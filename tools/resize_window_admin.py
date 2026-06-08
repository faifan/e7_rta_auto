"""以管理员权限运行，从 JSON 文件读参数后调整窗口大小。
用法：python resize_window_admin.py <json_path>
"""
import sys, json, ctypes, ctypes.wintypes, time

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(0)
except Exception:
    pass

class RECT(ctypes.Structure):
    _fields_ = [('left',ctypes.c_long),('top',ctypes.c_long),
                ('right',ctypes.c_long),('bottom',ctypes.c_long)]

def _get_physical_screen_w():
    """通过 GDI 获取真实物理像素宽（不受 DPI-unaware 影响）。"""
    hdc = ctypes.windll.gdi32.CreateDCW('DISPLAY', None, None, None)
    w = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES
    ctypes.windll.gdi32.DeleteDC(hdc)
    return w

def main():
    if len(sys.argv) != 2:
        print('用法: python resize_window_admin.py <json_path>')
        sys.exit(1)

    with open(sys.argv[1], encoding='utf-8') as f:
        args = json.load(f)

    hwnd     = args['hwnd']
    target_w = args['w']   # 物理目标宽（= profile 宽，即用户选的分辨率）
    target_h = args['h']

    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32

    if not u32.IsWindow(hwnd):
        print('hwnd 无效'); sys.exit(2)

    if u32.IsIconic(hwnd):
        u32.ShowWindow(hwnd, 9)
        time.sleep(0.5)

    wr = RECT(); u32.GetWindowRect(hwnd, ctypes.byref(wr))
    if wr.left < -10000 or wr.top < -10000:
        print('窗口不在当前桌面，请先切换到游戏窗口再重试')
        sys.exit(4)

    cr = RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr))
    if cr.right == 0 or cr.bottom == 0:
        print('客户区为空，请确保游戏窗口已显示')
        sys.exit(5)

    # 计算 DPI 缩放（GetDeviceCaps 返回真实物理宽，不受 DPI-unaware 影响）
    logical_sw  = u32.GetSystemMetrics(0)
    logical_sh  = u32.GetSystemMetrics(1)
    physical_sw = _get_physical_screen_w()
    dpi_scale   = physical_sw / logical_sw if logical_sw else 1.0

    # 用户要求：物理像素 = profile 尺寸，所以 DPI-unaware 目标 = profile ÷ dpi_scale
    virt_target_w = round(target_w / dpi_scale)
    virt_target_h = round(target_h / dpi_scale)

    print(f'逻辑屏幕: {logical_sw}x{logical_sh}  物理屏幕: {physical_sw}  DPI缩放: {dpi_scale:.3f}')
    print(f'目标物理: {target_w}x{target_h}  → DPI-unaware: {virt_target_w}x{virt_target_h}')

    os_deco_w = (wr.right - wr.left) - cr.right
    os_deco_h = (wr.bottom - wr.top) - cr.bottom
    outer_w = virt_target_w + os_deco_w
    outer_h = virt_target_h + os_deco_h

    u32.keybd_event(0x12, 0, 0, 0)
    u32.SetForegroundWindow(hwnd)
    u32.keybd_event(0x12, 0, 0x0002, 0)
    time.sleep(0.3)

    ret = u32.SetWindowPos(hwnd, 0, 0, 0, outer_w, outer_h, 0x0002 | 0x0004)
    err = k32.GetLastError()
    if not ret:
        print(f'SetWindowPos 失败 error={err}')
        input('按回车关闭...')
        sys.exit(3)

    time.sleep(0.3)

    # 修正 DPI 非整数缩放（如 1.5×）导致的 ±1 像素舍入误差
    cr2 = RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr2))
    diff_w = virt_target_w - cr2.right
    diff_h = virt_target_h - cr2.bottom
    if diff_w != 0 or diff_h != 0:
        u32.SetWindowPos(hwnd, 0, 0, 0,
                         outer_w + diff_w, outer_h + diff_h,
                         0x0002 | 0x0004)
        time.sleep(0.2)

    cr3 = RECT(); u32.GetClientRect(hwnd, ctypes.byref(cr3))
    phys_w = round(cr3.right * dpi_scale)
    phys_h = round(cr3.bottom * dpi_scale)
    print(f'成功：窗口物理尺寸已设为 {phys_w}x{phys_h}')
    time.sleep(3)
    sys.exit(0)

if __name__ == '__main__':
    main()
