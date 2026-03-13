import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


def _save_with_fallback(fig, ani, save_path, fps, max_px_width=3840, max_px_height=2160):
    """
    Try saving the animation as-is; if ffmpeg rejects the frame size (e.g., extremely wide figures),
    reduce DPI to fit within a reasonable pixel bound and retry. This changes nothing when no error occurs.
    """
    try:
        ani.save(save_path, writer='ffmpeg', fps=fps)
        return
    except Exception as e:
        try:
            w_in, h_in = fig.get_size_inches()
            dpi = fig.get_dpi()
            width_px = int(round(w_in * dpi))
            height_px = int(round(h_in * dpi))

            scale = min(max_px_width / max(width_px, 1), max_px_height / max(height_px, 1), 0.9)
            if scale >= 1.0:
                scale = 0.9

            new_dpi = max(1, int(dpi * scale))
            fig.set_dpi(new_dpi)

            w_px = int(round(fig.get_size_inches()[0] * fig.get_dpi()))
            h_px = int(round(fig.get_size_inches()[1] * fig.get_dpi()))
            if w_px % 2 != 0 or h_px % 2 != 0:
                fig.set_dpi(fig.get_dpi() + ((w_px % 2) or (h_px % 2)))

            ani.save(save_path, writer='ffmpeg', fps=fps)
            return
        except Exception:
            try:
                fig.set_dpi(50)
                ani.save(save_path, writer='ffmpeg', fps=fps)
                return
            except Exception:
                raise e

def make_saliency_map_video(concepts, saliency_maps, save_path, fps=4, color_map='inferno'):
    """
        For each concept, create a video using matplotlib where each frame is displayed as a heatmap.

        Inputs:
            concepts: List[str]
            saliency_maps: torch.Tensor of shape (num_concepts, num_frames, height, width)
    """
    num_concepts, num_frames, height, width = saliency_maps.shape

    ncols = 5
    nrows = max(1, math.ceil(num_concepts / ncols))
    cell_w, cell_h = 4, 4 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * cell_h))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.ravel().tolist()
    for ax in axes[num_concepts:]:
        ax.axis('off')

    def update(frame):
        for i in range(num_concepts):
            ax = axes[i]
            ax.clear()
            ax.set_title(concepts[i])
            heatmap = saliency_maps[i, frame, :, :].to(torch.float32).cpu().numpy()
            ax.imshow(
                heatmap,
                cmap=color_map,
                interpolation='nearest',
                vmin=saliency_maps.min(),
                vmax=saliency_maps.max()
            )

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    _save_with_fallback(fig, ani, save_path, fps)
    plt.close(fig)

def make_individual_videos(concepts, saliency_maps, save_dir, fps=4, color_map='inferno'):

    def make_individual_video(concept, saliency_map, save_path):

        h, w = saliency_map.shape[1:] 
        aspect_ratio = w / h
        fig, ax = plt.subplots(figsize=(5 * aspect_ratio, 5))
        fig.patch.set_visible(False) 

        def update(frame):
            ax.clear()
            ax.axis('off')  
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
            heatmap = saliency_map[frame, :, :].to(torch.float32).cpu().numpy()
            ax.imshow(
                heatmap, 
                cmap=color_map, 
                interpolation='nearest',
                vmin=saliency_map.min(),
                vmax=saliency_map.max()
            )

        ani = animation.FuncAnimation(fig, update, frames=saliency_map.shape[0], repeat=False)
        _save_with_fallback(fig, ani, save_path, fps)
        plt.close(fig)

    for i, concept in enumerate(concepts):
        saliency_map = saliency_maps[i]
        make_individual_video(concept, saliency_map, f"{save_dir}/{concept}_attention_video.mp4")