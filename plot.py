import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

def plot_layout_with_temperature(chiplets, temp_map, intpsize, *,
                                 title="Chiplet Layout + Temperature",
                                 cmap="coolwarm",  # 低温偏蓝，高温偏红
                                 vmin=None, vmax=None,
                                 rect_edgecolor="k", rect_facecolor="white",
                                 rect_alpha=0.25, linewidth=1.5,
                                 show_labels=True, save_path=None, color_bar = False):
    """
    将温度分布与chiplet布局融合显示。
    
    Parameters
    ----------
    chiplets : list[dict]
        形如 [{"x": x0, "y": y0, "w": w0, "h": h0, "name": "C0"}, ...]
        坐标为左下角 (x, y)，单位与 intpsize 一致。
    temp_map : np.ndarray
        形状为 (64, 64) 或 (H, W) 的温度网格。
    intpsize : tuple[float, float]
        (W, H)，布局板框大小（与 chiplet 坐标单位一致）。
    title : str
        图标题。
    cmap : str
        温度 colormap，默认 "coolwarm"（蓝-红）。
    vmin, vmax : float | None
        温度颜色范围；默认自动取 temp_map 的最小/最大。
    rect_edgecolor : str
        chiplet 边框颜色。
    rect_facecolor : str
        chiplet 填充颜色（会配合 alpha 半透明）。
    rect_alpha : float
        chiplet 填充透明度（0~1）。
    linewidth : float
        chiplet 边框线宽。
    show_labels : bool
        是否在矩形中标注 name。
    save_path : str | None
        若提供路径，则保存图像到该文件。
    """
    W, H = intpsize
    temp_map = np.asarray(temp_map)

    # 自动设定颜色范围（也可手工传入 vmin/vmax）
    if vmin is None: vmin = float(np.nanmin(temp_map))
    if vmax is None: vmax = float(np.nanmax(temp_map))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # 用 extent 把热图坐标映射到 [0,W]×[0,H]，origin='lower' 匹配左下角为原点
    im = ax.imshow(temp_map, origin='lower', extent=(0, W, 0, H),
                   cmap=cmap, norm=norm, interpolation='bilinear')  # bilinear 显得更平滑

    # 叠加 chiplet 矩形
    for i, c in enumerate(chiplets):
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        name = c.get("name", "")
        rect = Rectangle((x, y), w, h,
                         linewidth=linewidth,
                         edgecolor=rect_edgecolor,
                         facecolor=rect_facecolor,
                         alpha=rect_alpha)
        ax.add_patch(rect)

        if show_labels and name and i < 20:
            ax.text(x + w*0.5, y + h*0.5, name,
                    ha="center", va="center", fontsize=8, weight="bold",
                    color="black")

    # 轴设定
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"X $(\mu m)$")
    ax.set_ylabel(r"Y $(\mu m)$")
    # ax.set_title(title)

    # 颜色条
    if color_bar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation='vertical')
    cbar.set_label("T(°C)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# ===== 示例用法 =====
if __name__ == "__main__":
    for i in range(1, 11):
        case_id = f"{i}"
        case_name = f"Case{case_id}"
        gf_np = np.load(f"4fig/{case_name}_gf.npz")
        compact_np = np.load(f"4fig/{case_name}_compact.npz")
        gf_pred = gf_np['pred']
        ground_truth = gf_np['gt']
        compact_pred = compact_np['pred']
        sid = compact_np['sid']
        
        case_dir = f"./cases/{case_name}"
        intpsize_path = os.path.join(case_dir, f"{case_name}.intpsize")
        with open(intpsize_path) as f:
                line = f.readline().strip()
                width = height = float(line)
        intpsize = (width, height)  # 板框大小 W, H
        
        chiplets = []
        pl_path = os.path.join(case_dir, f"{case_name}_{sid}.pl")
        with open(pl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    x, y = float(parts[1]), float(parts[2])  # 左下角坐标(um)
                    w, h = float(parts[3]), float(parts[4])  # 宽高(um)
                    chiplets.append({"x": x, "y": y, "w": w, "h": h, "name": parts[0]})

        vmin = float(np.nanmin(ground_truth))
        vmax = float(np.nanmax(ground_truth))
        plot_layout_with_temperature(
            chiplets, ground_truth, intpsize,
            title="Layout + Thermal Map",
            cmap="jet",
            rect_facecolor="white",
            rect_alpha=0.3,
            linewidth=1.8,
            show_labels=True,
            save_path=f"4fig/{case_name}_ground_truth.png" , 
            # color_bar= True
        )
        plot_layout_with_temperature(
            chiplets, gf_pred, intpsize,
            title="Layout + Thermal Map",
            cmap="jet",
            rect_facecolor="white",
            rect_alpha=0.3,
            linewidth=1.8,
            show_labels=True,
            save_path=f"4fig/{case_name}_gf.png"  ,
            vmax=vmax,
            vmin=vmin,
        )
        plot_layout_with_temperature(
            chiplets, compact_pred, intpsize,
            title="Layout + Thermal Map",
            cmap="jet",
            rect_facecolor="white",
            rect_alpha=0.3,
            linewidth=1.8,
            show_labels=True,
            save_path=f"4fig/{case_name}_compact.png",
            color_bar = True ,
            vmax=vmax,
            vmin=vmin
        )
