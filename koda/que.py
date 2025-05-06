
from typing import IO, List, Dict, Any, Callable, Tuple, Optional, Union, cast
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import filetype
import os.path
from RangeSlider.RangeSlider import RangeSliderH
import matplotlib.image as mpimg
import tkinter.filedialog as tkfd
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backend_tools import ToolBase
from matplotlib.widgets import SpanSelector

#mpl.use('QtAgg')
plt.rcParams['toolbar'] = 'toolmanager'
def colormap_random_color() -> Tuple[float, float, float, float]:
    return plt.get_cmap('hsv')(np.random.rand())

class FSpanSelector(SpanSelector):
    def __init__(self, ax: plt.Axes, onselect_with_self: Callable[['FSpanSelector', float, float], None], 
                 id: int, *args: Any, **kwargs: Any):
        self.id: int = id
        super().__init__(ax, *args, **kwargs, onselect=lambda vmin, vmax: onselect_with_self(self, vmin, vmax))


def init(master: tk.Tk) -> None:
    def on_select(sel: FSpanSelector, xmin: float, xmax: float) -> None:
        scaled_xmin, scaled_xmax = xmin * scaling_factor, xmax * scaling_factor

        for s in spans:
            if s != sel and max(s.extents[0], xmin) < min(s.extents[1], xmax):
                move_r, move_l = s.extents[1] - xmin, xmax - s.extents[0]
                sel.extents = (xmin + move_r, xmax + move_r) if move_r <= move_l else (xmin - move_l, xmax - move_l)

        row_idx = sel.id
        data.loc[row_idx, ["Start", "End", "Time Diff"]] = [scaled_xmin, scaled_xmax, scaled_xmax - scaled_xmin]
        tree.item(row_idx, values=data.iloc[row_idx].tolist())

    def set_x_axis_scaling() -> None:
        nonlocal scaling_factor
        max_seconds = float(x_axis_input.get())
        if max_seconds <= 0:
            raise ValueError()
        scaling_factor = max_seconds / ax.get_xlim()[1]
        update_xticks()

    def update_xticks() -> None:
        x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=10)  # gen ticks
        scaled_ticks: List[float] = [tick * scaling_factor for tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in scaled_ticks])
        plt.draw()

    def edit_cell(e: tk.Event) -> None:
        region = tree.identify_region(e.x, e.y)
        c_id = tree.identify_column(e.x)
        r_id = int(tree.identify_row(e.y))
        c_idx = int(c_id[1:]) - 1
        print("Editing: ", r_id, c_id, c_idx)
        c_key: str = tree['columns'][c_idx]
        if c_key == "Pos": return
        print(c_idx)
        val: Any = tree.item(r_id, 'values')[c_idx]

        entry: tk.Entry = tk.Entry(tree)
        entry.place(x=e.x, y=e.y)
        entry.insert(0, val)
        entry.focus()

        def confirm(e: tk.Event) -> None:
            new_val: str = entry.get()
            row_idx = int(r_id)
            print(row_idx, c_key)
            data.at[row_idx, c_key] = float(new_val) if c_key in ['Start', 'End', 'Time Diff'] else new_val
            tree.item(r_id, values=data.iloc[row_idx].tolist())
            entry.destroy()

        entry.bind("<Return>", confirm)
        entry.bind("<FocusOut>", lambda e: entry.destroy())
        print(e)
        pass

    def add_row() -> None:
        print("add row")
        row_id = len(data)
        data.loc[row_id] = [row_id, 0, 0.2, 0.2, "WASTE", "LOC2"]
        refresh()

        s: FSpanSelector = FSpanSelector(plt.gca(), lambda o, vmin, vmax: on_select(o, vmin, vmax), row_id, direction="horizontal", useblit=True,
                        # props=dict(facecolor="dodgerblue"),
                        props=dict(facecolor=colormap_random_color(), alpha=0.9),
                            interactive=True, drag_from_anywhere=True, grab_range=8, ignore_event_outside=True)
        spans.append(s)
    
    def refresh() -> None:
        tree.delete(*tree.get_children())
        for i in data.index:
            tree.insert("", "end", iid=i, values=data.iloc[i].tolist())

    def update_slider(event: tk.Event) -> None:
        print(f"SMin={slider.getMin()}, Max={slider.getMax()}")

    def on_select_row(event: tk.Event) -> None:
        selected_item: Tuple[str, ...] = tree.selection()
        if not selected_item:
            return
        row_idx = int(selected_item[0])
        if row_idx < len(spans):
            for span in spans:
                span.set_props(alpha=0.9)
                span.update()
            spans[row_idx].set_props(alpha=0.6)
            spans[row_idx].update()

    def load_graph() -> None:
        f: Optional[IO] = tkfd.askopenfile(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
        if f is None:
            return
        df: pd.DataFrame = pd.read_csv(f)
        ax.clear()
        ax.plot(df.iloc[:, 0], df.iloc[:, 1])
        ax.set_xlim(left=0)
        plt.draw()
        set_x_axis_scaling()

    def load_image() -> None:
        f: Optional[IO] = tkfd.askopenfile()
        if f is None:
            return
        img: np.ndarray = plt.imread(f.name)
        ax.clear()
        ax.imshow(img, aspect='auto')
        ax.set_xlim(left=0)
        plt.draw()
        set_x_axis_scaling()

    def export_to_csv() -> None:
        file_path = tkfd.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        time_adjustment: float = float(waste_time_input.get())
        export_data: pd.DataFrame = data.copy().sort_values(by="Start").reset_index(drop=True)

        # insert waste rows between spans
        rows: List[Dict[str, Any]] = []
        pos_counter = 0
        for i in range(len(export_data)):
            current_row: Dict[str, Any] = export_data.iloc[i].to_dict()
            current_row["Pos"] = pos_counter
            rows.append(current_row)
            pos_counter += 1

            if i < len(export_data) - 1:
                current_end: float = export_data.iloc[i]["End"]
                next_start: float = export_data.iloc[i + 1]["Start"]
                if current_end < next_start:
                    waste_row: Dict[str, Any] = {
                        "Pos": pos_counter,
                        "Start": current_end,
                        "End": next_start - time_adjustment,
                        "Time Diff": (next_start - time_adjustment) - current_end,
                        "LOC1": "WASTE",
                        "LOC2": "WASTE"
                    }
                    rows.append(waste_row)
                    pos_counter += 1

        export_data_with_waste = pd.DataFrame(rows)
        export_data_with_waste.to_csv(file_path, index=False)
        print(f"exported to {file_path}")

    def on_click(event) -> None:
        if event.button != 3:
            return
        click_x: Optional[float] = event.xdata
        if click_x is None:
            return
        for span in spans[:]:
            x_min: float = span.extents[0]
            x_max: float = span.extents[1]
            if x_min <= click_x <= x_max:
                span.set_active(False)
                span.set_visible(False)
                span.update()
                spans.remove(span)

                row_idx: int = span.id
                data.drop(index=row_idx, inplace=True)

                data.reset_index(drop=True, inplace=True)
                data["Pos"] = data.index

                # update the IDs
                for i, s in enumerate(spans):
                    s.id = i

                refresh()
                break

    master = master
    data: pd.DataFrame = pd.DataFrame(columns=["Pos", "Start", "End", "Time Diff", "LOC1", "LOC2"])
    spans: List[FSpanSelector] = []

    master.grid_rowconfigure(0, weight=1)
    master.grid_columnconfigure(0, weight=1)
    master.grid_columnconfigure(1, weight=7)

    frame = tk.Frame(master)
    frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=10)

    tree = ttk.Treeview(frame, columns=data.columns.tolist(), show="headings")
    for col in tree["columns"]:
        tree.heading(col, text=col)

    style = ttk.Style()
    style.configure("Treeview.Heading", padding=(0, 5))

    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    button_frame = tk.Frame(frame)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    tk.Button(button_frame, text="Add Span", command=add_row).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Load Graph", command=load_graph).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Load Image", command=load_image).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Export to CSV", command=export_to_csv).pack(side=tk.LEFT, padx=5)

    scaling_frame = tk.Frame(button_frame)
    scaling_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(scaling_frame, text="Max Seconds:").pack(side=tk.LEFT)
    x_axis_input = tk.Entry(scaling_frame, width=8)
    x_axis_input.pack(side=tk.LEFT, padx=2)
    tk.Button(scaling_frame, text="Set", command=set_x_axis_scaling).pack(side=tk.LEFT)

    waste_time_frame = tk.Frame(button_frame)
    waste_time_frame.pack(side=tk.LEFT, padx=5)
    tk.Label(waste_time_frame, text="Waste Time Adjustment:").pack(side=tk.LEFT)
    waste_time_input = tk.Entry(waste_time_frame, width=8)
    waste_time_input.insert(0, "0.1")
    waste_time_input.pack(side=tk.LEFT, padx=2)

    tree.bind("<Double-1>", edit_cell)
    tree.bind("<<TreeviewSelect>>", on_select_row)

    plot_frame = tk.Frame(master)
    plot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(plt.figure(1), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, pady=5)
    tree.pack(side=tk.TOP, fill=tk.Y, expand=1, pady=5)

    slider_frame = tk.Frame(plot_frame)
    slider_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    slider: RangeSliderH = RangeSliderH(slider_frame, [tk.DoubleVar(value=0.2), tk.DoubleVar(value=0.8)], padX=18)
    slider.pack()
    slider.bind("<Motion>", update_slider)

    ax = plt.gca()

    canvas.mpl_connect("button_press_event", on_click)

    scaling_factor = 1.0

if __name__ == '__main__':
    root: tk.Tk = tk.Tk()
    app: None = init(root)
    root.mainloop()
#https://github.com/harshvinay752/RangeSlider
