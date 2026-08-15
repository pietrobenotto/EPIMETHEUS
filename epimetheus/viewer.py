import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
from typing import List, Optional

from IPython import get_ipython
from IPython.display import display, clear_output
import ipywidgets as widgets

from .core import StarCutouts


class StarReviewer:
    """
    Interactive reviewer for star cutouts.
    """

    def __init__(self, star_cutouts: StarCutouts, save_each_n=30, cmap: str = "magma", norm=None, figsize=(6, 5)):
        self.save_catalogue_func = star_cutouts.save_catalogue
        list_of_stars_to_check = star_cutouts.unknown_stars()
        self.stars_list = [star for i, star in enumerate(star_cutouts.stars) if i in list_of_stars_to_check]
        self.save_each_n = save_each_n
        self.upsample_factor = star_cutouts.upsample_factor
        self.cmap = cmap
        self.norm = norm
        self.figsize = figsize
        self.current_index = 0
        self.decisions: List[Optional[bool]] = [None] * len(self.stars_list)

        self._is_jupyter = get_ipython() is not None
        self._container = None
        self._out = None

        # Store references to prevent garbage collection (for non-Jupyter)
        self._fig = None
        self._ax = None
        self._cax = None  # Dedicated colorbar axes
        self._im = None   # Image object for updating
        self._cbar = None
        self._hline = None
        self._vline = None
        self._btn_back = None
        self._btn_good = None
        self._btn_bad = None
        self._btn_finish = None

    def run(self):
        if len(self.stars_list) == 0:
            print("No stars to review.")
            return

        if self._is_jupyter:
            self._render_jupyter()
        else:
            self._render_python()

    def apply_decisions(self):
        for item, decision in zip(self.stars_list, self.decisions):
            if decision is True:
                item.is_valid = True
                item.failure_reason = None
            elif decision is False:
                item.is_valid = False
                item.failure_reason = "Manually marked as bad"

        self.save_catalogue_func()

    def _set_decision(self, decision: bool):
        self.decisions[self.current_index] = decision

    def _next_star(self):
        if self.current_index > 0 and self.current_index % self.save_each_n == 0:
            self.apply_decisions()

        if self.current_index < len(self.stars_list) - 1:
            self.current_index += 1
            self._refresh()
        else:
            self._finish()

    def _prev_star(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._refresh()

    def _set_remaining_bad(self):
        for index in range(self.current_index, len(self.stars_list)):
            self.decisions[index] = False
        self._finish()

    def _finish(self):
        self.apply_decisions()

        if self._is_jupyter:
            with self._out:
                clear_output(wait=True)
                print("Review complete.")
                print("Decisions have been applied to the star objects.")
        else:
            plt.close(self._fig)
            print("Review complete.")
            print("Decisions have been applied to the star objects.")

    def _refresh(self):
        if self._is_jupyter:
            self._refresh_jupyter_plot()
        else:
            self._update_python_plot()

    # ------------------------------------------------------------------
    # JUPYTER IMPLEMENTATION
    # ------------------------------------------------------------------

    def _render_jupyter(self):
        self._out = widgets.Output()

        btn_back = widgets.Button(description="Back", button_style='', layout={'width': '100px'})
        btn_good = widgets.Button(description="Good", button_style='success', layout={'width': '100px'})
        btn_bad = widgets.Button(description="Bad", button_style='danger', layout={'width': '100px'})
        btn_finish = widgets.Button(description="Finish", button_style='warning', layout={'width': '100px'})

        btn_back.on_click(lambda _: self._prev_star())
        btn_good.on_click(lambda _: [self._set_decision(True), self._next_star()])
        btn_bad.on_click(lambda _: [self._set_decision(False), self._next_star()])
        btn_finish.on_click(lambda _: self._set_remaining_bad())

        controls = widgets.HBox([btn_back, btn_good, btn_bad, btn_finish])
        self._container = widgets.VBox([self._out, controls])

        display(self._container)
        self._refresh_jupyter_plot()

    def _refresh_jupyter_plot(self):
        with self._out:
            clear_output(wait=True)
            self._plot_current()
            plt.show()

    # ------------------------------------------------------------------
    # STANDARD PYTHON IMPLEMENTATION
    # ------------------------------------------------------------------

    def _render_python(self):
        """
        Create and show the Matplotlib review window for standard Python use.
        """
        from matplotlib.widgets import Button
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Create figure with explicit gridspec for better control
        self._fig = plt.figure(figsize=self.figsize)
        
        # Create main axes with room for buttons at bottom
        self._ax = self._fig.add_axes([0.1, 0.25, 0.65, 0.65])
        
        # Create dedicated colorbar axes (won't be removed, just updated)
        self._cax = self._fig.add_axes([0.78, 0.25, 0.03, 0.65])

        # Create button axes
        ax_back = self._fig.add_axes([0.10, 0.08, 0.12, 0.07])
        ax_good = self._fig.add_axes([0.26, 0.08, 0.12, 0.07])
        ax_bad = self._fig.add_axes([0.42, 0.08, 0.12, 0.07])
        ax_finish = self._fig.add_axes([0.74, 0.08, 0.12, 0.07])

        # Create buttons and store as instance attributes
        self._btn_back = Button(ax_back, 'Back', color='lightgray')
        self._btn_good = Button(ax_good, 'Good', color='lightgreen')
        self._btn_bad = Button(ax_bad, 'Bad', color='salmon')
        self._btn_finish = Button(ax_finish, 'Finish', color='khaki')

        self._btn_back.on_clicked(self._on_back_clicked)
        self._btn_good.on_clicked(self._on_good_clicked)
        self._btn_bad.on_clicked(self._on_bad_clicked)
        self._btn_finish.on_clicked(self._on_finish_clicked)

        # Initial plot (creates self._im, self._cbar, etc.)
        self._create_initial_plot()

        plt.show()

    def _on_back_clicked(self, event):
        self._prev_star()

    def _on_good_clicked(self, event):
        self._set_decision(True)
        self._next_star()

    def _on_bad_clicked(self, event):
        self._set_decision(False)
        self._next_star()

    def _on_finish_clicked(self, event):
        self._set_remaining_bad()

    def _create_initial_plot(self):
        """
        Create the initial plot elements that will be updated later.
        """
        item = self.stars_list[self.current_index]
        decision = self.decisions[self.current_index]

        # Compute norm
        if self.norm is None:
            norm = AsinhNorm(1e-5/(item.scale_factor*self.upsample_factor**2)**2, np.percentile(item.cutout_elaborated, 1))
        else:
            norm = self.norm

        # Create image
        self._im = self._ax.imshow(
            item.cutout_elaborated, 
            origin="lower", 
            cmap=self.cmap, 
            norm=norm
        )

        # Create colorbar in dedicated axes
        self._cbar = self._fig.colorbar(self._im, cax=self._cax)

        # Center crosshairs
        height, width = item.cutout_elaborated.shape
        center_y = height / 2.0 - 0.5
        center_x = width / 2.0 - 0.5
        self._hline, = self._ax.plot(
            [0, width-1], [center_y, center_y], 
            'w--', linewidth=0.8, alpha=0.2
        )
        self._vline, = self._ax.plot(
            [center_x, center_x], [0, height-1], 
            'w--', linewidth=0.8, alpha=0.2
        )

        # Set title
        self._update_title()
        self._update_xlabel()

    def _update_python_plot(self):
        """
        Update the existing figure without creating/removing axes.
        """
        item = self.stars_list[self.current_index]

        # Compute new norm
        if self.norm is None:
            norm = AsinhNorm(1e-5/(item.scale_factor*self.upsample_factor**2)**2, np.percentile(item.cutout_elaborated, 1))
        else:
            norm = self.norm

        # Update image data and norm
        self._im.set_data(item.cutout_elaborated)
        self._im.set_norm(norm)
        
        # Update image extent if shape changed
        height, width = item.cutout_elaborated.shape
        self._im.set_extent([-0.5, width - 0.5, -0.5, height - 0.5])

        # Update crosshairs
        center_y = height / 2.0 - 0.5
        center_x = width / 2.0 - 0.5
        self._hline.set_data([0, width-1], [center_y, center_y])
        self._vline.set_data([center_x, center_x], [0, height-1])

        # Update axes limits
        self._ax.set_xlim(-0.5, width - 0.5)
        self._ax.set_ylim(-0.5, height - 0.5)

        # Update colorbar (just refresh, no remove/recreate)
        self._cbar.update_normal(self._im)

        # Update title and labels
        self._update_title()
        self._update_xlabel()

        # Redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _update_title(self):
        """Update the title based on current state."""
        item = self.stars_list[self.current_index]
        decision = self.decisions[self.current_index]

        mag = getattr(item, "mag", None)
        if mag is not None:
            title = f"Star {self.current_index} | Mag: {mag:.2f}"
        else:
            title = f"Star {self.current_index}"

        decision_str = {True: "GOOD", False: "BAD", None: "UNDECIDED"}[decision]

        self._ax.set_title(
            f"{title}\nReview {self.current_index + 1}/{len(self.stars_list)} | Status: {decision_str}"
        )

    def _update_xlabel(self):
        """Update the xlabel based on current item."""
        item = self.stars_list[self.current_index]
        mag = getattr(item, "mag", None)
        if mag is not None:
            rotation = getattr(item, "rotation", None)
            if rotation is not None:
                self._ax.set_xlabel(f"rotation {rotation:.2f}")
            else:
                self._ax.set_xlabel("")
        else:
            self._ax.set_xlabel("")

    def _plot_current(self):
        """
        Plot the currently selected review item (for Jupyter).
        """
        item = self.stars_list[self.current_index]
        decision = self.decisions[self.current_index]

        fig, ax = plt.subplots(figsize=self.figsize)
        if self.norm is None:
            norm = AsinhNorm(1e-5/(item.scale_factor*self.upsample_factor**2)**2, np.percentile(item.cutout_elaborated, 1))
        else:
            norm = self.norm
        im = ax.imshow(item.cutout_elaborated, origin="lower", cmap=self.cmap, norm=norm)

        mag = getattr(item, "mag", None)
        if mag is not None:
            title = f"Star {self.current_index} | Mag: {mag:.2f}"
            if item.rotation is not None:
                ax.set_xlabel(f"rotation {item.rotation:.2f}")
        else:
            title = f"Star {self.current_index}"

        decision_str = {True: "GOOD", False: "BAD", None: "UNDECIDED"}[decision]

        height, width = item.cutout_elaborated.shape
        center_y = height / 2.0 - 0.5
        center_x = width / 2.0 - 0.5

        ax.axhline(center_y, color='w', linestyle='--', linewidth=0.8, alpha=0.2)
        ax.axvline(center_x, color='w', linestyle='--', linewidth=0.8, alpha=0.2)

        ax.set_title(
            f"{title}\nReview {self.current_index + 1}/{len(self.stars_list)} | Status: {decision_str}"
        )

        plt.colorbar(im)

        return fig