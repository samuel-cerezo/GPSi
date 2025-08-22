#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, math, argparse
from pyproj import Proj
from bagpy import bagreader

# ---------- utilidades ----------
def read_csv(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows

def get_first(d, keys, default=None, cast=float):
    for k in keys:
        if k in d and d[k] not in (None, "", "nan"):
            try:
                return cast(d[k])
            except Exception:
                try:
                    return float(d[k])
                except Exception:
                    return d[k]
    return default

def ts_ns_from_row(row):
    secs = get_first(row, ["header.stamp.secs","header.stamp.sec","stamp.secs","stamp.sec"], None, int)
    nsecs = get_first(row, ["header.stamp.nsecs","header.stamp.nsec","stamp.nsecs","stamp.nsec"], None, int)
    if secs is not None and nsecs is not None:
        return int(secs)*10**9 + int(nsecs)
    t = get_first(row, ["Time","time","timestamp","t"], None, float)
    return int(float(t)*1e9) if t is not None else None

def utm_proj_for(lat, lon):
    zone = int((lon + 180.0) / 6) + 1
    north = lat >= 0
    return Proj(proj="utm", zone=zone, ellps="WGS84", south=not north)

# ---------- script principal ----------
def main():
    ap = argparse.ArgumentParser(description="Exporta IMU, GPS y GT 4-DoF usando RTK como ground truth si existe (bagpy, sin ROS1).")
    ap.add_argument("bag", help="Ruta al .bag (ROS1)")
    ap.add_argument("--min-speed", type=float, default=0.2, help="Umbral [m/s] para estimar yaw desde velocidad cuando no hay rtk_yaw")
    args = ap.parse_args()

    bag_path = os.path.abspath(args.bag)
    seq_dir = os.path.dirname(bag_path)

    # salidas
    imu_out = os.path.join(seq_dir, "imu.csv")
    gps_out = os.path.join(seq_dir, "gps.csv")
    gps_utm_out = os.path.join(seq_dir, "gps_utm.csv")
    gt_out = os.path.join(seq_dir, "gt_4dof.csv")

    b = bagreader(bag_path)
    topics = set(b.topic_table["Topics"].tolist())

    # Preferencias IMU
    imu_topic = None
    for t in ["/livox/imu", "/dji_osdk_ros/imu"]:
        if t in topics: imu_topic = t; break

    # Fuentes “navegación” para gps.csv
    nav_pos_topic = None
    for t in ["/dji_osdk_ros/gps_position", "/dji_osdk_ros/rtk_position"]:
        if t in topics: nav_pos_topic = t; break

    nav_vel_topic = None
    for t in ["/dji_osdk_ros/gps_velocity", "/dji_osdk_ros/rtk_velocity"]:
        if t in topics: nav_vel_topic = t; break

    # Fuentes GT (priorizar RTK)
    rtk_pos_topic = "/dji_osdk_ros/rtk_position" if "/dji_osdk_ros/rtk_position" in topics else None
    rtk_vel_topic = "/dji_osdk_ros/rtk_velocity" if "/dji_osdk_ros/rtk_velocity" in topics else None
    rtk_yaw_topic = "/dji_osdk_ros/rtk_yaw" if "/dji_osdk_ros/rtk_yaw" in topics else None

    print(f"IMU: {imu_topic or 'no encontrada'}")
    print(f"GPS (nav) pos: {nav_pos_topic or 'no encontrada'} | vel: {nav_vel_topic or 'no encontrada'}")
    print(f"RTK (GT) pos: {rtk_pos_topic or 'no encontrado'} | vel: {rtk_vel_topic or 'no encontrado'} | yaw: {rtk_yaw_topic or 'no encontrado'}")

    # -------- exportar CSV brutos con bagpy --------
    def export_topic(t):
        if not t: return None
        try: return b.message_by_topic(t)
        except Exception as e:
            print(f"[WARN] No se pudo exportar {t}: {e}")
            return None

    imu_raw = export_topic(imu_topic)
    nav_pos_raw = export_topic(nav_pos_topic)
    nav_vel_raw = export_topic(nav_vel_topic)

    rtk_pos_raw = export_topic(rtk_pos_topic)
    rtk_vel_raw = export_topic(rtk_vel_topic)
    rtk_yaw_raw = export_topic(rtk_yaw_topic)

    # -------- IMU → imu.csv (EuRoC) --------
    if imu_raw and os.path.exists(imu_raw):
        rows = read_csv(imu_raw)
        with open(imu_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(['#timestamp [ns]', 'wx [rad/s]', 'wy [rad/s]', 'wz [rad/s]',
                        'ax [m/s^2]', 'ay [m/s^2]', 'az [m/s^2]'])
            n = 0
            for r in rows:
                ts = ts_ns_from_row(r)
                if ts is None: continue
                wx = get_first(r, ["angular_velocity.x","angular_velocity_x"])
                wy = get_first(r, ["angular_velocity.y","angular_velocity_y"])
                wz = get_first(r, ["angular_velocity.z","angular_velocity_z"])
                ax = get_first(r, ["linear_acceleration.x","linear_acceleration_x"])
                ay = get_first(r, ["linear_acceleration.y","linear_acceleration_y"])
                az = get_first(r, ["linear_acceleration.z","linear_acceleration_z"])
                if None in (wx,wy,wz,ax,ay,az): continue
                w.writerow([int(ts), float(wx), float(wy), float(wz), float(ax), float(ay), float(az)])
                n += 1
        print(f"imu.csv ✓  ({n} filas)  -> {imu_out}")
    else:
        print("imu.csv ✗  (no se exportó IMU)")

    # -------- GPS de navegación → gps.csv / gps_utm.csv --------
    gps_buffer = []  # (ts, lat, lon, alt, vx, vy, vz)

    # cache de velocidades
    vel_cache = {}
    if nav_vel_raw and os.path.exists(nav_vel_raw):
        for r in read_csv(nav_vel_raw):
            ts = ts_ns_from_row(r)
            if ts is None: continue
            vx = get_first(r, ["vector.x","vector_x","twist.linear.x","x"], 0.0)
            vy = get_first(r, ["vector.y","vector_y","twist.linear.y","y"], 0.0)
            vz = get_first(r, ["vector.z","vector_z","twist.linear.z","z"], 0.0)
            vel_cache[int(ts)] = (float(vx), float(vy), float(vz))

    # posición
    if nav_pos_raw and os.path.exists(nav_pos_raw):
        rows = read_csv(nav_pos_raw)
        for r in rows:
            ts = ts_ns_from_row(r)
            if ts is None: continue
            lat = get_first(r, ["latitude","lat"])
            lon = get_first(r, ["longitude","lon"])
            alt = get_first(r, ["altitude","alt","position.altitude"])
            if None in (lat, lon, alt): continue
            vx = vy = vz = 0.0
            if int(ts) in vel_cache:
                vx, vy, vz = vel_cache[int(ts)]
            gps_buffer.append((int(ts), float(lat), float(lon), float(alt), float(vx), float(vy), float(vz)))
    else:
        print("gps.csv ✗  (no se exportó posición de navegación)")

    gps_buffer.sort(key=lambda x: x[0])

    if gps_buffer:
        with open(gps_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(['#timestamp [ns]', 'latitude [deg]', 'longitude [deg]', 'altitude [m]',
                        'vx [m/s]', 'vy [m/s]', 'vz [m/s]'])
            for row in gps_buffer:
                w.writerow(row)
        print(f"gps.csv ✓  ({len(gps_buffer)} filas)  -> {gps_out}")

        # gps_utm.csv
        lat0, lon0 = gps_buffer[0][1], gps_buffer[0][2]
        proj = utm_proj_for(lat0, lon0)
        with open(gps_utm_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(['timestamp [ns]','x [m]','y [m]','z [m]','vx [m/s]','vy [m/s]','vz [m/s]'])
            for ts, lat, lon, alt, vx, vy, vz in gps_buffer:
                x, y = proj(lon, lat)
                w.writerow([ts, x, y, alt, vx, vy, vz])
        print(f"gps_utm.csv ✓ -> {gps_utm_out}")
    else:
        print("gps_utm.csv ✗  (no hay GPS de navegación)")

    # -------- GT 4-DoF (RTK como ground truth) --------
    gt_rows = []  # (ts, x, y, z, yaw)

    # Preferentemente RTK:
    use_rtk = rtk_pos_raw and os.path.exists(rtk_pos_raw)
    if use_rtk:
        print("GT: usando RTK como ground truth.")
        # caches
        rtk_vel_cache = {}
        rtk_yaw_cache = {}

        if rtk_vel_raw and os.path.exists(rtk_vel_raw):
            for r in read_csv(rtk_vel_raw):
                ts = ts_ns_from_row(r)
                if ts is None: continue
                vx = get_first(r, ["vector.x","vector_x","twist.linear.x","x"], 0.0)
                vy = get_first(r, ["vector.y","vector_y","twist.linear.y","y"], 0.0)
                vz = get_first(r, ["vector.z","vector_z","twist.linear.z","z"], 0.0)
                rtk_vel_cache[int(ts)] = (float(vx), float(vy), float(vz))

        if rtk_yaw_raw and os.path.exists(rtk_yaw_raw):
            for r in read_csv(rtk_yaw_raw):
                ts = ts_ns_from_row(r)
                if ts is None: continue
                data = get_first(r, ["data"], None, float)
                if data is None: continue
                # asume grados a rad (ajusta si tu CSV documenta otra escala)
                rtk_yaw_cache[int(ts)] = math.radians(float(data))

        rtk_rows = read_csv(rtk_pos_raw)
        proj_gt = None
        for r in rtk_rows:
            ts = ts_ns_from_row(r)
            if ts is None: continue
            lat = get_first(r, ["latitude","lat"])
            lon = get_first(r, ["longitude","lon"])
            alt = get_first(r, ["altitude","alt","position.altitude"])
            if None in (lat, lon, alt): continue

            if proj_gt is None:
                proj_gt = utm_proj_for(lat, lon)
            x, y = proj_gt(lon, lat)

            yaw = None
            if int(ts) in rtk_yaw_cache:
                yaw = rtk_yaw_cache[int(ts)]
            else:
                # yaw desde velocidad si hay y supera umbral
                vx = vy = 0.0
                if int(ts) in rtk_vel_cache:
                    vx, vy, _ = rtk_vel_cache[int(ts)]
                if math.hypot(vx, vy) >= args.min_speed:
                    yaw = math.atan2(vy, vx)

            if yaw is not None:
                gt_rows.append((int(ts), x, y, float(alt), float(yaw)))

    else:
        # Fallback: GPS de navegación como GT (degradado)
        print("GT: RTK no disponible -> usando gps_position+gps_velocity (DEGRADADO).")
        if gps_buffer:
            proj_gt = utm_proj_for(gps_buffer[0][1], gps_buffer[0][2])
            for ts, lat, lon, alt, vx, vy, vz in gps_buffer:
                if math.hypot(vx, vy) >= args.min_speed:
                    yaw = math.atan2(vy, vx)
                    x, y = proj_gt(lon, lat)
                    gt_rows.append((ts, x, y, alt, yaw))

    gt_rows.sort(key=lambda r: r[0])
    if gt_rows:
        with open(gt_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp [ns]","x_utm [m]","y_utm [m]","z [m]","yaw [rad]"])
            w.writerows(gt_rows)
        print(f"gt_4dof.csv ✓  ({len(gt_rows)} poses) -> {gt_out}")
    else:
        print("gt_4dof.csv ✗  (no se pudo generar GT 4-DoF)")

if __name__ == "__main__":
    main()
