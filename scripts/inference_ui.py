"""
实例分割推理 UI（基于 ins_inference_encoded 底层）
用法：python scripts/inference_ui.py
"""
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, MODEL_DIR
from ins_inference_encoded import run_inference


class InferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实例分割推理工具（编码图）")
        self.root.geometry("1400x800")
        self.root.minsize(1100, 650)

        self.orig_image = None
        self.viz_image = None      # 可视化大图（raw + processed 2x4）
        self.result_data = None     # run_inference 返回的结果

        self._build_ui()

    def _build_ui(self):
        # ===== 顶部：参数区 =====
        param_frame = ttk.LabelFrame(self.root, text="参数设置", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # 图片
        ttk.Label(param_frame, text="图片:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.img_path = tk.StringVar()
        ttk.Entry(param_frame, textvariable=self.img_path, width=80).grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(param_frame, text="选择图片", command=self._select_image).grid(row=0, column=2)

        # 权重
        ttk.Label(param_frame, text="权重:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.model_path = tk.StringVar(value=MODEL_DIR)
        ttk.Entry(param_frame, textvariable=self.model_path, width=80).grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Button(param_frame, text="选择目录", command=self._select_model).grid(row=1, column=2)

        # 保存
        ttk.Label(param_frame, text="保存:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.save_path = tk.StringVar()
        ttk.Entry(param_frame, textvariable=self.save_path, width=80).grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Button(param_frame, text="选择目录", command=self._select_save).grid(row=2, column=2)

        # 参数行
        params_row = ttk.Frame(param_frame)
        params_row.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))

        # 置信度阈值
        ttk.Label(params_row, text="置信度:").pack(side=tk.LEFT)
        self.threshold = tk.DoubleVar(value=0.5)
        ttk.Scale(params_row, from_=0.1, to=0.95, variable=self.threshold,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 2))
        self.threshold_label = ttk.Label(params_row, text="0.50", width=4)
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 15))
        self.threshold.trace_add("write", self._update_threshold_label)

        # mask 阈值
        ttk.Label(params_row, text="mask阈值:").pack(side=tk.LEFT)
        self.mask_threshold = tk.DoubleVar(value=0.5)
        ttk.Scale(params_row, from_=0.1, to=0.95, variable=self.mask_threshold,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 2))
        self.mask_thresh_label = ttk.Label(params_row, text="0.50", width=4)
        self.mask_thresh_label.pack(side=tk.LEFT, padx=(0, 15))
        self.mask_threshold.trace_add("write", lambda *_: self.mask_thresh_label.config(text=f"{self.mask_threshold.get():.2f}"))

        # 面投票阈值
        ttk.Label(params_row, text="投票阈值:").pack(side=tk.LEFT)
        self.min_ratio = tk.DoubleVar(value=0.5)
        ttk.Scale(params_row, from_=0.1, to=0.95, variable=self.min_ratio,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=(0, 2))
        self.min_ratio_label = ttk.Label(params_row, text="0.50", width=4)
        self.min_ratio_label.pack(side=tk.LEFT, padx=(0, 15))
        self.min_ratio.trace_add("write", lambda *_: self.min_ratio_label.config(text=f"{self.min_ratio.get():.2f}"))

        # 按钮
        ttk.Button(params_row, text="开始推理", command=self._run_inference).pack(side=tk.RIGHT, padx=10)
        ttk.Button(params_row, text="批量推理", command=self._run_batch).pack(side=tk.RIGHT, padx=5)

        param_frame.columnconfigure(1, weight=1)

        # ===== 状态栏 =====
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, padx=10, pady=(0, 5))

        # ===== 中间：Notebook（原图 / 可视化结果 / 检测详情） =====
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Tab 1: 原图 + 可视化
        img_frame = ttk.Frame(notebook)
        notebook.add(img_frame, text="图片对比")

        self.orig_canvas = tk.Canvas(img_frame, bg="#2b2b2b")
        self.orig_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        self.viz_canvas = tk.Canvas(img_frame, bg="#2b2b2b")
        self.viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        self.orig_canvas.bind("<Configure>", lambda e: self._redraw("orig"))
        self.viz_canvas.bind("<Configure>", lambda e: self._redraw("viz"))

        # Tab 2: 检测详情表格
        detail_frame = ttk.Frame(notebook)
        notebook.add(detail_frame, text="检测详情")

        cols = ("实例ID", "面ID", "面类型", "类别ID", "类别名", "置信度", "面积", "投票占比")
        self.tree = ttk.Treeview(detail_frame, columns=cols, show="headings", height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _update_threshold_label(self, *_):
        self.threshold_label.config(text=f"{self.threshold.get():.2f}")

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="选择编码图",
            filetypes=[("PNG", "*.png"), ("All", "*.*")]
        )
        if path:
            self.img_path.set(path)
            self._show_orig(path)
            if not self.save_path.get():
                self.save_path.set(os.path.join(os.path.dirname(path), "inference_results"))

    def _select_model(self):
        path = filedialog.askdirectory(title="选择模型权重目录")
        if path:
            self.model_path.set(path)

    def _select_save(self):
        path = filedialog.askdirectory(title="选择保存目录")
        if path:
            self.save_path.set(path)

    def _show_orig(self, path):
        img = Image.open(path).convert("RGB")
        self.orig_image = img
        self._redraw("orig")

    def _redraw(self, which):
        canvas = self.orig_canvas if which == "orig" else self.viz_canvas
        image = self.orig_image if which == "orig" else self.viz_image
        if image is None:
            return

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        iw, ih = image.size
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        resized = image.resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=tk_img)
        canvas._tk_img = tk_img

    def _populate_table(self, class_map):
        """填充检测详情表格"""
        self.tree.delete(*self.tree.get_children())
        if not class_map:
            return
        for inst_id, info in class_map.items():
            self.tree.insert("", tk.END, values=(
                inst_id,
                info.get("face_id", "-"),
                info.get("face_type_name", "-"),
                info.get("class_id", "-"),
                info.get("class_name", "-"),
                f"{info.get('score', 0):.4f}",
                info.get("area", "-"),
                f"{info.get('vote_ratio', '-')}" if isinstance(info.get("vote_ratio"), (int, float)) else "-",
            ))

    def _run_inference(self):
        img_path = self.img_path.get()
        model_dir = self.model_path.get()
        save_dir = self.save_path.get()

        if not img_path or not os.path.exists(img_path):
            messagebox.showerror("错误", "请先选择一张图片")
            return
        if not model_dir or not os.path.exists(model_dir):
            messagebox.showerror("错误", "模型权重目录不存在")
            return
        if not save_dir:
            save_dir = os.path.join(os.path.dirname(img_path), "inference_results")
            self.save_path.set(save_dir)

        self.status_var.set("推理中（加载模型 + 提取面 + 推理 + 后处理）...")
        self.root.update()

        try:
            result = run_inference(
                image_path=img_path,
                model_dir=model_dir,
                output_dir=save_dir,
                threshold=self.threshold.get(),
                mask_threshold=self.mask_threshold.get(),
                min_ratio=self.min_ratio.get(),
                min_face_area=10,
                device_name="auto",
            )
        except Exception as e:
            messagebox.showerror("推理失败", str(e))
            self.status_var.set("推理失败")
            return

        self.result_data = result

        # 加载可视化图
        prefix = os.path.splitext(os.path.basename(img_path))[0]
        viz_path = os.path.join(save_dir, f"{prefix}_visualization.png")
        if os.path.exists(viz_path):
            self.viz_image = Image.open(viz_path).convert("RGB")
            self._redraw("viz")

        # 填充详情表格（用 processed 结果，如果没有则用 raw）
        class_map = result.get("processed_class_map") or result.get("raw_class_map") or {}
        self._populate_table(class_map)

        # 统计
        n_faces = len(result.get("face_masks", {}))
        n_raw = len(result.get("raw_class_map", {}))
        n_proc = len(result.get("processed_class_map", {}))
        self.status_var.set(
            f"推理完成 | 检测到 {n_faces} 个面 → 原始 {n_raw} 个实例 → 后处理 {n_proc} 个实例 | "
            f"结果已保存到 {save_dir}"
        )

    def _run_batch(self):
        img_path = self.img_path.get()
        model_dir = self.model_path.get()

        if not img_path:
            messagebox.showerror("错误", "请先选择图片或图片目录")
            return
        if not model_dir or not os.path.exists(model_dir):
            messagebox.showerror("错误", "模型权重目录不存在")
            return

        input_dir = os.path.dirname(img_path) if os.path.isfile(img_path) else img_path
        save_dir = self.save_path.get() or os.path.join(input_dir, "inference_results")

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png',))]
        if not image_files:
            messagebox.showerror("错误", f"目录中没有 PNG 图片: {input_dir}")
            return

        self.status_var.set(f"批量推理中... 0/{len(image_files)}")
        self.root.update()

        count = 0
        errors = 0
        for i, fname in enumerate(image_files):
            fpath = os.path.join(input_dir, fname)
            self.status_var.set(f"批量推理中... {i + 1}/{len(image_files)} - {fname}")
            self.root.update()
            try:
                run_inference(
                    image_path=fpath,
                    model_dir=model_dir,
                    output_dir=save_dir,
                    threshold=self.threshold.get(),
                    mask_threshold=self.mask_threshold.get(),
                    min_ratio=self.min_ratio.get(),
                    min_face_area=10,
                    device_name="auto",
                )
                count += 1
            except Exception as e:
                errors += 1
                print(f"[Skip] {fname}: {e}")

        self.status_var.set(f"批量推理完成：成功 {count}/{len(image_files)}，失败 {errors}，保存到 {save_dir}")
        messagebox.showinfo("完成",
                            f"批量推理完成\n成功: {count}/{len(image_files)}\n失败: {errors}\n保存位置: {save_dir}")


def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
