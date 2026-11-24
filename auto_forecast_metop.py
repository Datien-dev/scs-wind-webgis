import os
import io
import math
import json
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "model_assets")
OUT_DATA_DIR = os.path.join(ROOT_DIR, "docs", "data")
os.makedirs(OUT_DATA_DIR, exist_ok=True)

AUX_VARS_C = [
    "t2m_raw", "sst_raw", "msl_raw", "swh_raw", "pp1d_raw",
    "delta_t", "ri_b", "z_over_L", "blh",
    "dp_dx", "dp_dy",
    "shww", "shts", "wave_age", "cos_wind_swell", "sin_wind_swell", "charnock",
    "sst_grad_mag", "dsst_downwind", "dsst_crosswind", "sst_anom",
]

STATIONS = [
    ("Bạch Long Vĩ", 20.13, 107.72),
    ("Hoàng Sa", 16.53, 111.62),
    ("Đà Nẵng", 16.07, 108.22),
    ("Phú Quý", 10.52, 108.93),
    ("Trường Sa", 8.64, 111.92),
    ("Côn Đảo", 8.68, 106.60),
    ("Hải Phòng", 20.85, 106.68),
    ("TP. Hồ Chí Minh", 10.77, 106.70),
]

# Bảng màu liên tục giống Zoom Earth
ZOOM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "zoom_earth",
    [
        (0.00, "#4b2e83"),
        (0.16, "#3b528b"),
        (0.32, "#21918c"),
        (0.48, "#5ec962"),
        (0.64, "#fde725"),
        (0.78, "#f89540"),
        (0.90, "#cc4778"),
        (1.00, "#900c3f"),
    ],
)


# ==============================================================================
# 1. KIẾN TRÚC DA-DSC-UNet ĐỘC LẬP (CHẠY NHẸ TRÊN CPU/GPU)
# ==============================================================================
class DualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        k = max(3, int(abs((math.log2(channels) + 1) / 2)))
        k = k if k % 2 == 1 else k + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.lambda_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        gap = x.mean(dim=(-2, -1), keepdim=False).unsqueeze(1)
        a_c = torch.sigmoid(self.eca_conv(gap)).transpose(1, 2).unsqueeze(-1)
        m_avg = x.mean(dim=1, keepdim=True)
        m_max, _ = x.max(dim=1, keepdim=True)
        a_s = torch.sigmoid(self.spatial_conv(torch.cat([m_avg, m_max], dim=1)))
        lam = torch.sigmoid(self.lambda_gate)
        return x * (lam * a_c + (1.0 - lam) * a_s) + x


class DSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            DualAttentionBlock(out_ch),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ASPPBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.b2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.b3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False)
        self.b4 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=4, dilation=4, bias=False)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        f5 = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x), f5], dim=1))


class DualBranchDADSCUNet(nn.Module):
    def __init__(self, wind_in_ch=36, aux_in_ch=27, out_steps=4):
        super().__init__()
        self.out_steps = out_steps
        self.has_aux = aux_in_ch > 0
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DSCBlock(wind_in_ch, 64)
        self.enc2 = DSCBlock(64, 128)
        self.enc3 = DSCBlock(128, 256)
        self.bot_wind = DSCBlock(256, 256)
        if self.has_aux:
            self.aux_stem = nn.Sequential(nn.Conv2d(aux_in_ch, 16, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, True))
            self.aux_enc1 = DSCBlock(16, 32, dropout_p=0.10)
            self.aux_enc2 = DSCBlock(32, 64, dropout_p=0.10)
            self.aux_enc3 = DSCBlock(64, 128, dropout_p=0.10)
            self.bot_aux = DSCBlock(128, 256, dropout_p=0.15)
            self.gate_conv = nn.Conv2d(512, 256, kernel_size=1, bias=True)
            self.surf_refine = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, True))
            self.surf_scale = nn.Parameter(torch.tensor(0.05))
        self.aspp = ASPPBottleneck(256, 256)
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = DSCBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DSCBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DSCBlock(128, 64)
        self.head = nn.Conv2d(64, out_steps * 2, kernel_size=1)

    def forward(self, x_wind, x_aux, uv_t0):
        e1 = self.enc1(x_wind)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bw = self.bot_wind(self.pool(e3))
        if self.has_aux:
            a1 = self.aux_enc1(self.aux_stem(x_aux))
            a2 = self.aux_enc2(self.pool(a1))
            a3 = self.aux_enc3(self.pool(a2))
            ba = self.bot_aux(self.pool(a3))
            b_fused = bw + torch.sigmoid(self.gate_conv(torch.cat([bw, ba], dim=1))) * ba
        else:
            b_fused = bw
        b = self.aspp(b_fused)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        if self.has_aux:
            d1 = d1 + torch.tanh(self.surf_scale) * self.surf_refine(torch.cat([d1, a1], dim=1))
        delta_uv = self.head(d1).view(-1, self.out_steps, 2, 64, 72)
        return uv_t0.unsqueeze(1) + delta_uv


# ==============================================================================
# 2. TỰ ĐỘNG TẢI VỆT QUÉT VỆ TINH METOP-B/C ASCAT NRT (EUMETSAT DATA STORE)
# ==============================================================================
def fuse_metop_ascat_nrt(sel_times, u_in, v_in, obs_mask, lat, lon):
    """
    Kết nối EUMETSAT Data Store (eumdac) lấy vệt quét MetOp-B/C ASCAT L2 Coastal 25km
    trong 6 ngày qua và ghép trực tiếp vào các ô lưới tương ứng trên Biển Đông.
    """
    key = os.environ.get("EUMETSAT_KEY", "").strip()
    secret = os.environ.get("EUMETSAT_SECRET", "").strip()
    if not key or not secret:
        print("  [MetOp NRT] Chưa cấu hình EUMETSAT_KEY/SECRET -> Dùng trường nền Open-Meteo ECMWF IFS.")
        return u_in, v_in, obs_mask, "Open-Meteo ECMWF IFS + Marine (NRT)"

    try:
        import eumdac
        import xarray as xr

        token = eumdac.AccessToken((key, secret))
        datastore = eumdac.DataStore(token)
        # Bộ sưu tập chuẩn ASCAT Winds 25km Coastal trên MetOp-B/C
        collection = datastore.get_collection("EO:EUM:DAT:METOP:OAS025")
        dt_start = (sel_times[0] - pd.Timedelta(hours=6)).to_pydatetime()
        dt_end = (sel_times[-1] + pd.Timedelta(hours=6)).to_pydatetime()

        products = list(collection.search(dtstart=dt_start, dtend=dt_end, bbox="102,5,118,23"))
        if not products:
            print("  [MetOp NRT] Không tìm thấy vệt quét mới trong khung giờ -> Dùng nền Open-Meteo.")
            return u_in, v_in, obs_mask, "Open-Meteo ECMWF IFS + Marine (NRT)"

        fused_pixels = 0
        lat_min, lon_min = float(lat.min()), float(lon.min())
        dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            for prod in products[:24]:  # Giới hạn tối đa 24 vệt quét mới nhất để chạy nhanh
                try:
                    with prod.open() as fsrc:
                        raw_bytes = fsrc.read()
                    nc_file = None
                    if raw_bytes[:2] == b"PK":
                        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                            ncs = [n for n in zf.namelist() if n.endswith(".nc")]
                            if ncs:
                                nc_file = zf.extract(ncs[0], tmpdir)
                    else:
                        nc_file = os.path.join(tmpdir, "swath.nc")
                        with open(nc_file, "wb") as fout:
                            fout.write(raw_bytes)

                    if not nc_file:
                        continue

                    with xr.open_dataset(nc_file) as ds_sw:
                        s_lat = ds_sw["lat"].values.ravel()
                        s_lon = ds_sw["lon"].values.ravel()
                        s_lon = np.where(s_lon > 180, s_lon - 360.0, s_lon)
                        wspd = ds_sw["wind_speed"].values.ravel()
                        wdir = ds_sw["wind_dir"].values.ravel()
                        qflag = ds_sw["wvc_quality_flag"].values.ravel() if "wvc_quality_flag" in ds_sw else np.zeros_like(wspd)
                        t_sw = pd.to_datetime(prod.sensing_start).tz_localize(None)

                    # Tìm khung 12h gần nhất trong cửa sổ 6 ngày
                    diffs_h = np.abs((sel_times - t_sw).total_seconds() / 3600.0)
                    t_idx = int(np.argmin(diffs_h))
                    if diffs_h[t_idx] > 6.0:
                        continue

                    valid = (
                        (s_lat >= 5.0) & (s_lat <= 23.0) &
                        (s_lon >= 102.0) & (s_lon <= 118.0) &
                        np.isfinite(wspd) & np.isfinite(wdir) & (qflag == 0)
                    )
                    if not np.any(valid):
                        continue

                    iy = np.clip(np.round((s_lat[valid] - lat_min) / dlat).astype(int), 0, len(lat) - 1)
                    ix = np.clip(np.round((s_lon[valid] - lon_min) / dlon).astype(int), 0, len(lon) - 1)
                    rad = np.deg2rad(wdir[valid])
                    u_sat = -wspd[valid] * np.sin(rad)
                    v_sat = -wspd[valid] * np.cos(rad)

                    u_in[t_idx, iy, ix] = u_sat
                    v_in[t_idx, iy, ix] = v_sat
                    obs_mask[t_idx, iy, ix] = 1.0
                    fused_pixels += len(iy)
                except Exception:
                    continue

        if fused_pixels > 0:
            print(f"  [MetOp NRT OK] Đã đồng hóa thành công {fused_pixels:,} điểm đo vệ tinh MetOp-B/C ASCAT!")
            return u_in, v_in, obs_mask, f"MetOp-B/C ASCAT NRT ({fused_pixels:,} pts) + ECMWF IFS"
    except Exception as e:
        print(f"  [MetOp NRT Fallback] Lỗi kết nối EUMETSAT ({e}) -> Dùng nền Open-Meteo.")

    return u_in, v_in, obs_mask, "Open-Meteo ECMWF IFS + Marine (NRT)"


# ==============================================================================
# 3. TẢI DỮ LIỆU NỀN 6 NGÀY & TÍNH 27 KÊNH VẬT LÝ
# ==============================================================================
def fetch_inputs_6days(lat_target, lon_target, sea_mask_2d, depth_2d, dist_coast_2d):
    lats_c = np.linspace(5.0, 23.0, 15)
    lons_c = np.linspace(102.0, 118.0, 13)
    grid_la, grid_lo = np.meshgrid(lats_c, lons_c, indexing="ij")
    flat_la = ",".join([f"{x:.2f}" for x in grid_la.ravel()])
    flat_lo = ",".join([f"{x:.2f}" for x in grid_lo.ravel()])

    url_w = (
        f"https://api.open-meteo.com/v1/forecast?latitude={flat_la}&longitude={flat_lo}"
        f"&past_days=7&forecast_days=1"
        f"&hourly=wind_speed_10m,wind_direction_10m,temperature_2m,surface_pressure,boundary_layer_height"
        f"&wind_speed_unit=ms&timezone=UTC"
    )
    url_m = (
        f"https://marine-api.open-meteo.com/v1/marine?latitude={flat_la}&longitude={flat_lo}"
        f"&past_days=7&forecast_days=1"
        f"&hourly=wave_height,wave_period,wind_wave_height,swell_wave_height,swell_wave_direction,sea_surface_temperature"
        f"&timezone=UTC"
    )
    resp_w = requests.get(url_w, timeout=60).json()
    resp_m = requests.get(url_m, timeout=60).json()

    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    times_all = pd.to_datetime(resp_w[0]["hourly"]["time"])
    sel_idx = np.where(times_all.hour.isin([0, 12]) & (times_all <= now_utc))[0][-12:]
    sel_times = times_all[sel_idx]

    ny_c, nx_c, T = len(lats_c), len(lons_c), len(sel_idx)
    arrs = {k: np.zeros((T, ny_c, nx_c), dtype=np.float32) for k in ["u", "v", "t2m", "msl", "blh", "swh", "pp1d", "shww", "shts", "mdts", "sst"]}

    for idx_pt in range(ny_c * nx_c):
        iy, ix = divmod(idx_pt, nx_c)
        hw = resp_w[idx_pt]["hourly"]
        hm = resp_m[idx_pt]["hourly"] if isinstance(resp_m, list) and idx_pt < len(resp_m) else {}
        for t_i, h_pos in enumerate(sel_idx):
            ws = float(hw["wind_speed_10m"][h_pos] or 0.0)
            wd = float(hw["wind_direction_10m"][h_pos] or 0.0)
            arrs["u"][t_i, iy, ix] = -ws * np.sin(np.deg2rad(wd))
            arrs["v"][t_i, iy, ix] = -ws * np.cos(np.deg2rad(wd))
            arrs["t2m"][t_i, iy, ix] = float(hw["temperature_2m"][h_pos] or 26.0) + 273.15
            arrs["msl"][t_i, iy, ix] = float(hw["surface_pressure"][h_pos] or 1012.0) * 100.0
            arrs["blh"][t_i, iy, ix] = float(hw["boundary_layer_height"][h_pos] or 600.0)
            arrs["swh"][t_i, iy, ix] = float((hm.get("wave_height", [1.0] * (h_pos + 1))[h_pos]) or 1.0)
            arrs["pp1d"][t_i, iy, ix] = float((hm.get("wave_period", [6.0] * (h_pos + 1))[h_pos]) or 6.0)
            arrs["shww"][t_i, iy, ix] = float((hm.get("wind_wave_height", [0.7] * (h_pos + 1))[h_pos]) or 0.7)
            arrs["shts"][t_i, iy, ix] = float((hm.get("swell_wave_height", [0.5] * (h_pos + 1))[h_pos]) or 0.5)
            arrs["mdts"][t_i, iy, ix] = float((hm.get("swell_wave_direction", [45.0] * (h_pos + 1))[h_pos]) or 45.0)
            sst_v = hm.get("sea_surface_temperature", [27.5] * (h_pos + 1))[h_pos]
            arrs["sst"][t_i, iy, ix] = float(sst_v if sst_v is not None else 27.5) + 273.15

    target_la, target_lo = np.meshgrid(lat_target, lon_target, indexing="ij")
    pts = np.stack([target_la.ravel(), target_lo.ravel()], axis=-1)

    def interp_fine(a3d):
        out = np.zeros((T, len(lat_target), len(lon_target)), dtype=np.float32)
        for ti in range(T):
            out[ti] = RegularGridInterpolator((lats_c, lons_c), a3d[ti], bounds_error=False, fill_value=None)(pts).reshape(len(lat_target), len(lon_target))
        return out

    f = {k: interp_fine(v) for k, v in arrs.items()}
    obs_mask = np.broadcast_to(sea_mask_2d[None, :, :].astype(np.float32), f["u"].shape).copy()

    # Đồng hóa vệt quét vệ tinh MetOp-B/C ASCAT NRT nếu có
    u_in, v_in, obs_mask, source_label = fuse_metop_ascat_nrt(sel_times, f["u"], f["v"], obs_mask, lat_target, lon_target)

    W = np.hypot(u_in, v_in) + 1e-4
    delta_t = f["t2m"] - f["sst"]
    g = 9.80665
    ri_b = np.clip((g / np.maximum(f["t2m"], 270.0)) * delta_t * 10.0 / (W ** 2 + 0.25), -5.0, 2.0)
    z_over_L = np.clip(np.where(ri_b < 0, 10.0 * ri_b / (1.0 - 3.0 * ri_b + 1e-5), 10.0 * ri_b / (1.0 + 5.0 * ri_b + 1e-5)), -10.0, 5.0)

    dx_km = 27.75 * np.cos(np.deg2rad(target_la))[None, :, :]
    dp_dx = np.gradient(f["msl"], axis=2) / (dx_km + 1e-4)
    dp_dy = np.gradient(f["msl"], axis=1) / 27.75

    omega = 2.0 * np.pi / np.maximum(f["pp1d"], 2.0)
    cp = (g / omega) * np.sqrt(np.tanh((omega ** 2 / g) * np.maximum(depth_2d[None, :, :], 5.0)))
    wave_age = np.clip(W / np.maximum(cp, 1.0), 0.05, 3.0)

    wind_dir_to = np.arctan2(u_in, v_in)
    swell_dir_to = np.deg2rad((f["mdts"] + 180.0) % 360.0)
    cos_ws, sin_ws = np.cos(wind_dir_to - swell_dir_to), np.sin(wind_dir_to - swell_dir_to)
    charnock = np.clip(0.018 * (wave_age ** 0.5), 0.008, 0.06)

    dsst_dx = np.gradient(f["sst"], axis=2) / (dx_km + 1e-4)
    dsst_dy = np.gradient(f["sst"], axis=1) / 27.75
    sst_grad_mag = np.hypot(dsst_dx, dsst_dy)
    dsst_downwind = (dsst_dx * u_in + dsst_dy * v_in) / W
    dsst_crosswind = (dsst_dx * (-v_in) + dsst_dy * u_in) / W
    sst_anom = f["sst"] - (27.8 + 273.15)

    w_win = np.stack([np.transpose(u_in, (0, 2, 1)), np.transpose(v_in, (0, 2, 1)), np.transpose(obs_mask, (0, 2, 1))], axis=1)
    doy = sel_times.dayofyear.values
    t_feats = [np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32), np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32), (sel_times.hour.values == 12).astype(np.float32)]
    aux_list = [np.broadcast_to(t[:, None, None], (T, 64, 72)).copy() for t in t_feats]
    aux_list += [np.broadcast_to(np.transpose(s.astype(np.float32), (1, 0))[None], (T, 64, 72)).copy() for s in [sea_mask_2d, dist_coast_2d, depth_2d]]

    dyn_map = {
        "t2m_raw": f["t2m"], "sst_raw": f["sst"], "msl_raw": f["msl"], "swh_raw": f["swh"], "pp1d_raw": f["pp1d"],
        "delta_t": delta_t, "ri_b": ri_b, "z_over_L": z_over_L, "blh": f["blh"], "dp_dx": dp_dx, "dp_dy": dp_dy,
        "shww": f["shww"], "shts": f["shts"], "wave_age": wave_age, "cos_wind_swell": cos_ws, "sin_wind_swell": sin_ws, "charnock": charnock,
        "sst_grad_mag": sst_grad_mag, "dsst_downwind": dsst_downwind, "dsst_crosswind": dsst_crosswind, "sst_anom": sst_anom,
    }
    aux_list += [np.transpose(dyn_map[v].astype(np.float32), (0, 2, 1)) for v in AUX_VARS_C]
    a_win = np.stack(aux_list, axis=1)
    return sel_times, w_win, a_win, source_label


def save_heatmap_png(path, w_2d, sea_mask):
    arr = np.where(sea_mask, w_2d, np.nan)
    arr = np.flipud(arr)  # North-up cho Leaflet ImageOverlay
    norm = mcolors.Normalize(vmin=0.0, vmax=18.0, clip=True)
    rgba = ZOOM_CMAP(norm(arr))
    rgba[np.isnan(arr), 3] = 0.0
    rgba[~np.isnan(arr), 3] = 0.80
    plt.imsave(path, rgba, format="png")


def save_velocity_json(path, u_2d, v_2d, sea_mask, lat, lon, ref_time_str):
    u_ns = np.flipud(np.where(sea_mask, u_2d, 0.0))
    v_ns = np.flipud(np.where(sea_mask, v_2d, 0.0))
    header = {
        "parameterCategory": 2, "refTime": ref_time_str,
        "lo1": float(lon.min()), "la1": float(lat.max()),
        "lo2": float(lon.max()), "la2": float(lat.min()),
        "dx": abs(float(lon[1] - lon[0])), "dy": abs(float(lat[1] - lat[0])),
        "nx": int(len(lon)), "ny": int(len(lat)),
    }
    payload = [
        {"header": {**header, "parameterNumber": 2, "parameterNumberName": "U-component_of_wind"}, "data": np.round(u_ns, 2).ravel().tolist()},
        {"header": {**header, "parameterNumber": 3, "parameterNumberName": "V-component_of_wind"}, "data": np.round(v_ns, 2).ravel().tolist()},
    ]
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, separators=(",", ":"))


@torch.no_grad()
def main():
    stats = np.load(os.path.join(ASSETS_DIR, "norm_stats.npz"))
    lat, lon = stats["lat"], stats["lon"]
    sea_mask = stats["sea_mask"] > 0
    depth, dist_coast = stats["depth"], stats["dist_coast"]
    w_mean, w_std = stats["w_mean"], stats["w_std"]
    a_mean, a_std = stats["a_mean"], stats["a_std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchDADSCUNet(wind_in_ch=36, aux_in_ch=27, out_steps=4).to(device)
    state = torch.load(os.path.join(ASSETS_DIR, "best_da_dsc_unet_config_C.pth"), map_location=device)
    md = model.state_dict()
    md.update({k: v for k, v in state.items() if k in md and md[k].shape == v.shape})
    model.load_state_dict(md)
    model.eval()

    sel_times, w_win, a_win, source_label = fetch_inputs_6days(lat, lon, sea_mask, depth, dist_coast)
    t0 = sel_times[-1]

    x_w = ((w_win - w_mean) / w_std).reshape(1, -1, 64, 72).astype(np.float32)
    a_n = np.clip((a_win - a_mean) / a_std, -5.0, 5.0).astype(np.float32)
    x_a = (0.7 * a_n[-1] + 0.3 * a_n[-2])[None]
    uv0 = w_win[-1, :2][None]

    pred = model(torch.from_numpy(x_w).to(device), torch.from_numpy(x_a).to(device), torch.from_numpy(uv0).to(device)).cpu().numpy()[0]

    # Ghép mốc t0 (0h) + 4 mốc dự báo (+12h..+48h) thành 5 điểm neo -> Nội suy PCHIP từng 3 giờ!
    u_anchor = np.concatenate([np.transpose(uv0[0, 0], (1, 0))[None], np.transpose(pred[:, 0], (0, 2, 1))], axis=0)
    v_anchor = np.concatenate([np.transpose(uv0[0, 1], (1, 0))[None], np.transpose(pred[:, 1], (0, 2, 1))], axis=0)

    anchor_hours = np.array([0, 12, 24, 36, 48], dtype=np.float32)
    target_hours = np.arange(0, 49, 3, dtype=int)  # 17 khung: 0h, +3h, +6h, ..., +48h

    u_3h = PchipInterpolator(anchor_hours, u_anchor, axis=0)(target_hours)
    v_3h = PchipInterpolator(anchor_hours, v_anchor, axis=0)(target_hours)
    w_3h = np.hypot(u_3h, v_3h)

    manifest = {
        "updated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "t0_utc": t0.strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": source_label,
        "bounds": [[float(lat.min()) - 0.125, float(lon.min()) - 0.125], [float(lat.max()) + 0.125, float(lon.max()) + 0.125]],
        "steps": [],
    }

    for i, lh in enumerate(target_hours):
        ft_utc = t0 + pd.Timedelta(hours=int(lh))
        ft_vn = ft_utc + pd.Timedelta(hours=7)  # Giờ Việt Nam (UTC+7)
        png_name = f"wind_{lh}h.png"
        vel_name = f"velocity_{lh}h.json"

        save_heatmap_png(os.path.join(OUT_DATA_DIR, png_name), w_3h[i], sea_mask)
        save_velocity_json(os.path.join(OUT_DATA_DIR, vel_name), u_3h[i], v_3h[i], sea_mask, lat, lon, ft_utc.strftime("%Y-%m-%d %H:%M:%S"))

        st_list = []
        for s_name, s_la, s_lo in STATIONS:
            iy = int(np.argmin(np.abs(lat - s_la)))
            ix = int(np.argmin(np.abs(lon - s_lo)))
            uu, vv, ww = float(u_3h[i, iy, ix]), float(v_3h[i, iy, ix]), float(w_3h[i, iy, ix])
            st_list.append({
                "name": s_name, "lat": s_la, "lon": s_lo,
                "w_ms": round(ww, 1), "dir_deg": round(float(np.rad2deg(np.arctan2(uu, vv)) % 360.0), 0),
            })

        manifest["steps"].append({
            "lead_h": int(lh),
            "label_vn": ft_vn.strftime("%d/%m %H:%M (VN)"),
            "label_utc": ft_utc.strftime("%d %b %H:%M UTC"),
            "is_anchor": bool(lh % 12 == 0),
            "heatmap_png": f"data/{png_name}",
            "velocity_json": f"data/{vel_name}",
            "max_wind_ms": round(float(np.nanmax(np.where(sea_mask, w_3h[i], np.nan))), 1),
            "stations": st_list,
        })

    with open(os.path.join(OUT_DATA_DIR, "manifest.json"), "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, ensure_ascii=False, indent=2)
    print(f">>> HOÀN TẤT DỰ BÁO 17 KHUNG THỜI ĐIỂM (0h..+48h)! Nguồn: {source_label}")


if __name__ == "__main__":
    main()