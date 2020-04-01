#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graphical user interface for Instaseis.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from PySide2 import QtGui, QtCore
from PySide2.QtWidgets import QApplication
import pyqtgraph as pg

from glob import iglob
import imp
import inspect
from mpl_toolkits.basemap import Basemap
import numpy as np
from obspy.imaging.mopad_wrapper import beach
from obspy import geodetics
from obspy.taup import TauPyModel
import os
import sys

from instaseis import open_db, Source, Receiver, FiniteSource

# Default to antialiased drawing.
pg.setConfigOptions(antialias=True, foreground=(50, 50, 50), background=None)

# Initialize model once.
tau_model = TauPyModel(model="ak135")

# Most generic way to get the data folder path.
DATA = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "data",
)


def compile_and_import_ui_files():
    """
    Automatically compiles all .ui files found in the same directory as the
    application py file.
    They will have the same name as the .ui files just with a .py extension.

    Needs to be defined in the same file as function loading the gui as it
    modifies the globals to be able to automatically import the created py-ui
    files. Its just very convenient.
    """
    directory = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    for filename in iglob(os.path.join(directory, "*.ui")):
        ui_file = filename
        py_ui_file = os.path.splitext(ui_file)[0] + os.path.extsep + "py"
        if not os.path.exists(py_ui_file) or (
            os.path.getmtime(ui_file) >= os.path.getmtime(py_ui_file)
        ):

            # No more function in pyside2 so we'll just call the built in tool
            # directly.
            import PySide2 as ref_mod

            pyside_dir = os.path.dirname(ref_mod.__file__)

            exe = os.path.join(pyside_dir, "uic")
            cmd = f'{exe} -g python --output="{py_ui_file}" "{ui_file}"'

            print("Compiling ui file: %s" % ui_file)
            print(f"Executing: '{cmd}'")
            os.system(cmd)
            print("Done")

        # Import the (compiled) file.
        try:
            import_name = os.path.splitext(os.path.basename(py_ui_file))[0]
            globals()[import_name] = imp.load_source(import_name, py_ui_file)
        except ImportError as e:
            print("Error importing %s" % py_ui_file)
            print(e.message)


class Window(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        # Injected by the compile_and_import_ui_files() function.
        self.ui = qt_window.Ui_MainWindow()  # NOQA
        self.ui.setupUi(self)

        label = {"z": "vertical", "e": "east", "n": "north"}
        for component in ["z", "n", "e"]:
            p = getattr(self.ui, "%s_graph" % component)
            p.setLabel("left", "Displacement", units="m")
            p.setLabel("bottom", "Time since event", units="s")

            p.setTitle(label[component].capitalize() + " component")

        self.ui.n_graph.setXLink(self.ui.e_graph)
        self.ui.n_graph.setXLink(self.ui.z_graph)
        self.ui.e_graph.setXLink(self.ui.z_graph)
        self.ui.n_graph.setYLink(self.ui.e_graph)
        self.ui.n_graph.setYLink(self.ui.z_graph)
        self.ui.e_graph.setYLink(self.ui.z_graph)

        # Set some random mt at startup.
        m_rr = 4.71e17
        m_tt = 3.81e15
        m_pp = -4.74e17
        m_rt = 3.99e16
        m_rp = -8.05e16
        m_tp = -1.23e17
        self.ui.m_rr.setValue(m_rr)
        self.ui.m_tt.setValue(m_tt)
        self.ui.m_pp.setValue(m_pp)
        self.ui.m_rt.setValue(m_rt)
        self.ui.m_rp.setValue(m_rp)
        self.ui.m_tp.setValue(m_tp)

        self.instaseis_db = None
        self.finite_source = None
        self.st_copy = None

        self.plot_map()
        self.plot_mt()
        self.update()

    @property
    def focmec(self):
        if self.ui.source_tab.currentIndex() == 0:
            return [
                float(self.ui.m_rr.value()),
                float(self.ui.m_tt.value()),
                float(self.ui.m_pp.value()),
                float(self.ui.m_rt.value()),
                float(self.ui.m_rp.value()),
                float(self.ui.m_tp.value()),
            ]
        elif self.ui.source_tab.currentIndex() == 1:
            source = Source.from_strike_dip_rake(
                latitude=float(self.ui.source_latitude.value()),
                longitude=float(self.ui.source_longitude.value()),
                depth_in_m=float(self.source_depth) * 1000.0,
                strike=float(self.ui.strike_slider.value()),
                dip=float(self.ui.dip_slider.value()),
                rake=float(self.ui.rake_slider.value()),
                M0=1e16,
            )
            return [
                source.m_rr,
                source.m_tt,
                source.m_pp,
                source.m_rt,
                source.m_rp,
                source.m_tp,
            ]

    def plot_mt(self):
        self.mpl_mt_figure = self.ui.mt_fig.fig
        self.mpl_mt_ax = self.mpl_mt_figure.add_axes([0.0, 0.0, 1.0, 1.0])
        self.mpl_mt_ax.set_axis_off()
        self.mpl_mt_figure.patch.set_alpha(0.0)
        self.mpl_mt_figure.set_facecolor("None")

        self._draw_mt()

    def _draw_mt(self):
        if not hasattr(self, "mpl_mt_ax"):
            return

        try:
            self.bb.remove()
        except Exception:
            pass

        fm = self.focmec
        fm = [_i / 1e16 for _i in fm]
        self.bb = beach(fm, xy=(0, 0), width=200, linewidth=1, facecolor="red")
        self.mpl_mt_ax.add_collection(self.bb)
        self.mpl_mt_ax.set_xlim(-105, 105)
        self.mpl_mt_ax.set_ylim(-105, 105)
        self.mpl_mt_figure.canvas.draw()

    def plot_mt_finite(self):
        self.mpl_mt_finite_figure = self.ui.mt_fig_finite.fig
        self.mpl_mt_finite_ax = self.mpl_mt_finite_figure.add_axes(
            [0.0, 0.0, 1.0, 1.0]
        )
        self.mpl_mt_finite_ax.set_axis_off()
        self.mpl_mt_finite_figure.patch.set_alpha(0.0)
        self.mpl_mt_finite_figure.set_facecolor("None")

        self._draw_mt_finite()

    def _draw_mt_finite(self):
        if not hasattr(self, "mpl_mt_finite_ax"):
            return

        try:
            self.bb_finite.remove()
        except Exception:
            pass

        self.bb_finite = beach(
            self.finite_source.CMT.tensor / 1e16,
            xy=(0, 0),
            width=200,
            linewidth=1,
            facecolor="red",
        )
        self.mpl_mt_finite_ax.add_collection(self.bb_finite)
        self.mpl_mt_finite_ax.set_xlim(-105, 105)
        self.mpl_mt_finite_ax.set_ylim(-105, 105)
        self.mpl_mt_finite_figure.canvas.draw()

    def plot_cmt_sliprate(self):
        fig = self.ui.cmt_sliprate.fig
        fig.clf()
        fig.set_facecolor("white")
        ax = fig.add_axes([0.05, 0.2, 0.9, 0.8], frameon=False)
        ax.get_xaxis().tick_bottom()
        ax.axes.get_yaxis().set_visible(False)

        nsamp = len(self.finite_source.CMT.sliprate)
        time = np.linspace(
            0, self.finite_source.CMT.dt * nsamp, nsamp, endpoint=False
        )

        ax.plot(time, self.finite_source.CMT.sliprate)
        ax.set_xlim(0.0, self.finite_source.rupture_duration * 2.0)
        fig.canvas.draw()

    def plot_map(self):
        self.mpl_map_figure = self.ui.map_fig.fig

        # if hasattr(self, 'mpl_map_ax'):
        #     self.mpl_map_ax.clear()

        self.mpl_map_ax = self.mpl_map_figure.add_axes(
            [0.01, 0.01, 0.98, 0.98]
        )
        self.mpl_map_ax.set_title(
            "Left click: Set Receiver; Right click: Set " "Source"
        )

        self.map = Basemap(
            projection="moll", lon_0=0, resolution="c", ax=self.mpl_map_ax
        )

        self.map.drawmapboundary(fill_color="#cccccc")

        # Define planet radii and images paths
        radii_planets = {
            "Titan": 2575e3,
            "Europa": 1561e3,
            "Enceladus": 252e3,
            "Ganymede": 2631e3,
            "Mars": 3389.5e3,
            "Venus": 6051.8e3,
            "Earth": 6371e3,
        }
        imfiles = {
            "Titan": "titan_radar_colored_800.jpg",
            "Europa": "europa_comp_800.jpg",
            "Enceladus": "enceladus_jpl_800.jpg",
            "Ganymede": "ganymede_usgs_800.jpg",
            "Mars": "mola_texture_shifted_800.jpg",
            "Venus": "venus_magellan_800.jpg",
            "Earth": "earth_marble_ng_800.jpg",
            "default": "grid.png",
        }

        imfile = os.path.join(DATA, imfiles["default"])
        if self.instaseis_db:
            for key, value in radii_planets.items():
                if abs(self.instaseis_db.info.planet_radius - value) < 10e3:
                    imfile = os.path.join(DATA, imfiles[key])

        self.map.warpimage(image=imfile, zorder=0)

        self.mpl_map_figure.patch.set_alpha(0.0)

        self.mpl_map_figure.canvas.mpl_connect(
            "button_press_event", self._on_map_mouse_click_event
        )
        self.mpl_map_figure.canvas.draw()

    def _on_map_mouse_click_event(self, event):
        if None in (event.xdata, event.ydata):
            return
        # Get map coordinates by the inverse transform.
        lng, lat = self.map(event.xdata, event.ydata, inverse=True)
        # Left click: set receiver
        if event.button == 1:
            self.ui.receiver_longitude.setValue(lng)
            self.ui.receiver_latitude.setValue(lat)
        # Right click: set event
        elif event.button == 3 and self.ui.finsource_tab.currentIndex() == 0:
            self.ui.source_longitude.setValue(lng)
            self.ui.source_latitude.setValue(lat)

    def _plot_event(self):
        if self.ui.finsource_tab.currentIndex() == 0:
            s = self.source
            lng, lat = s.longitude, s.latitude
        elif self.ui.finsource_tab.currentIndex() == 1:
            s = self.finite_source
            s.find_hypocenter()
            lng, lat = s.hypocenter_longitude, s.hypocenter_latitude

        try:
            if (
                self.__event_map_obj.longitude == lng
                and self.__event_map_obj.latitude == lat
            ):
                return
        except AttributeError:
            pass

        try:
            self.__event_map_obj.remove()
        except AttributeError:
            pass

        x1, y1 = self.map(lng, lat)
        self.__event_map_obj = self.map.scatter(
            x1, y1, s=300, zorder=10, color="yellow", marker="*", edgecolor="k"
        )
        self.__event_map_obj.longitude = lng
        self.__event_map_obj.latitude = lat
        self.mpl_map_figure.canvas.draw()

    def _plot_receiver(self):
        r = self.receiver
        lng, lat = r.longitude, r.latitude
        try:
            if (
                self.__receiver_map_obj.longitude == lng
                and self.__receiver_map_obj.latitude == lat
            ):
                return
        except AttributeError:
            pass

        try:
            self.__receiver_map_obj.remove()
        except AttributeError:
            pass

        x1, y1 = self.map(lng, lat)
        self.__receiver_map_obj = self.map.scatter(
            x1, y1, s=170, zorder=10, color="red", marker="v", edgecolor="k"
        )
        self.__receiver_map_obj.longitude = lng
        self.__receiver_map_obj.latitude = lat
        self.mpl_map_figure.canvas.draw()

    def _plot_bg_receivers(self):
        try:
            self.__bg_receivers_map_obj.remove()
        except AttributeError:
            pass

        xl = []
        yl = []
        for r in self.receivers:
            lng, lat = r.longitude, r.latitude
            x1, y1 = self.map(lng, lat)
            xl.append(x1)
            yl.append(y1)

        self.__bg_receivers_map_obj = self.map.scatter(
            xl,
            yl,
            s=100,
            zorder=5,
            color="k",
            marker="v",
            edgecolor="gray",
            alpha=0.3,
        )
        self.mpl_map_figure.canvas.draw()

    @property
    def source(self):
        fm = self.focmec
        return Source(
            latitude=float(self.ui.source_latitude.value()),
            longitude=float(self.ui.source_longitude.value()),
            depth_in_m=float(self.source_depth) * 1000.0,
            m_rr=fm[0],
            m_tt=fm[1],
            m_pp=fm[2],
            m_rt=fm[3],
            m_rp=fm[4],
            m_tp=fm[5],
        )

    @property
    def receiver(self):
        return Receiver(
            latitude=float(self.ui.receiver_latitude.value()),
            longitude=float(self.ui.receiver_longitude.value()),
        )

    def update(self, force=False):

        try:
            self._plot_receiver()
            self._plot_event()
        except AttributeError:
            return

        if (
            not bool(self.ui.auto_update_check_box.checkState())
            and self.ui.finsource_tab.currentIndex() == 1
            and not force
            and self.st_copy is None
        ):
            return

        components = ["z", "n", "e"]
        components_map = {0: ("Z", "N", "E"), 1: ("Z", "R", "T")}

        components_choice = int(self.ui.components_combo.currentIndex())

        label_map = {
            0: {"z": "vertical", "n": "north", "e": "east"},
            1: {"z": "vertical", "n": "radial", "e": "transverse"},
        }

        for component in components:
            p = getattr(self.ui, "%s_graph" % component)
            p.setTitle(
                label_map[components_choice][component].capitalize()
                + " component"
            )

        if self.ui.finsource_tab.currentIndex() == 0:
            src_latitude = self.source.latitude
            src_longitude = self.source.longitude
            src_depth_in_m = self.source.depth_in_m
        else:
            src_latitude = self.finite_source.hypocenter_latitude
            src_longitude = self.finite_source.hypocenter_longitude
            src_depth_in_m = self.finite_source.hypocenter_depth_in_m

        rec = self.receiver
        try:
            # Grab resampling settings from the UI.
            if bool(self.ui.resample_check_box.checkState()):
                dt = float(self.ui.resample_factor.value())
                dt = self.instaseis_db.info.dt / dt
            else:
                dt = None
            if self.ui.finsource_tab.currentIndex() == 0:
                st = self.instaseis_db.get_seismograms(
                    source=self.source,
                    receiver=self.receiver,
                    dt=dt,
                    components=components_map[components_choice],
                )
            elif (
                not bool(self.ui.auto_update_check_box.checkState())
                and self.ui.finsource_tab.currentIndex() == 1
                and not force
            ):
                st = self.st_copy.copy()
            else:
                prog_diag = QtGui.QProgressDialog(
                    "Calculating", "Cancel", 0, len(self.finite_source), self
                )
                prog_diag.setWindowModality(QtCore.Qt.WindowModal)
                prog_diag.setMinimumDuration(0)

                def get_prog_fct():
                    def set_value(value, count):
                        prog_diag.setValue(value)
                        if prog_diag.wasCanceled():
                            return True

                    return set_value

                prog_diag.setValue(0)
                st = self.instaseis_db.get_seismograms_finite_source(
                    sources=self.finite_source,
                    receiver=self.receiver,
                    dt=dt,
                    components=("Z", "N", "E"),
                    progress_callback=get_prog_fct(),
                )
                prog_diag.setValue(len(self.finite_source))
                if not st:
                    return

                baz = geodetics.gps2dist_azimuth(
                    self.finite_source.CMT.latitude,
                    self.finite_source.CMT.longitude,
                    rec.latitude,
                    rec.longitude,
                )[2]
                self.st_copy = st.copy()
                st.rotate("NE->RT", baz)
                st += self.st_copy
                self.st_copy = st.copy()

            if self.ui.finsource_tab.currentIndex() == 1 and bool(
                self.ui.plot_CMT_check_box.checkState()
            ):
                st_cmt = self.instaseis_db.get_seismograms(
                    source=self.finite_source.CMT,
                    receiver=self.receiver,
                    dt=dt,
                    components=components_map[components_choice],
                    reconvolve_stf=True,
                    remove_source_shift=False,
                )
            else:
                st_cmt = None

            # check filter values from the UI
            zp = bool(self.ui.zero_phase_check_box.checkState())
            if bool(self.ui.lowpass_check_box.checkState()):
                try:
                    freq = 1.0 / float(self.ui.lowpass_period.value())
                    st.filter("lowpass", freq=freq, zerophase=zp)
                    if st_cmt is not None:
                        st_cmt.filter("lowpass", freq=freq, zerophase=zp)
                except ZeroDivisionError:
                    # this happens when typing in the lowpass_period box
                    pass

            if bool(self.ui.highpass_check_box.checkState()):
                try:
                    freq = 1.0 / float(self.ui.highpass_period.value())
                    st.filter("highpass", freq=freq, zerophase=zp)
                    if st_cmt is not None:
                        st_cmt.filter("highpass", freq=freq, zerophase=zp)
                except ZeroDivisionError:
                    # this happens when typing in the highpass_period box
                    pass

        except AttributeError:
            return

        if bool(self.ui.tt_times.checkState()):
            great_circle_distance = geodetics.locations2degrees(
                src_latitude, src_longitude, rec.latitude, rec.longitude
            )
            self.tts = tau_model.get_travel_times(
                source_depth_in_km=src_depth_in_m / 1000.0,
                distance_in_degree=great_circle_distance,
            )

        for ic, component in enumerate(components):
            plot_widget = getattr(self.ui, "%s_graph" % component.lower())
            plot_widget.clear()
            tr = st.select(component=components_map[components_choice][ic])[0]
            times = tr.times()
            plot_widget.plot(times, tr.data, pen="k")
            plot_widget.ptp = tr.data.ptp()
            if st_cmt is not None:
                tr = st_cmt.select(
                    component=components_map[components_choice][ic]
                )[0]
                times = tr.times()
                plot_widget.plot(times, tr.data, pen="r")

            if bool(self.ui.tt_times.checkState()):
                tts = []
                for tt in self.tts:
                    if tt.time >= times[-1]:
                        continue
                    tts.append(tt)
                    if tt.name[0].lower() == "p":
                        pen = "#008c2866"
                    else:
                        pen = "#95000066"
                    plot_widget.addLine(x=tt.time, pen=pen, z=-10)
                self.tts = tts
        self.set_info()

    @QtCore.Slot()
    def on_select_folder_button_released(self):
        pwd = os.getcwd()
        self.folder = str(
            QtGui.QFileDialog.getExistingDirectory(
                self, "Choose Directory", pwd
            )
        )
        if not self.folder:
            return
        self.instaseis_db = open_db(self.folder)

        # Adjust depth slider to the DB.
        max_rad = self.instaseis_db.info.max_radius / 1e3
        min_rad = self.instaseis_db.info.min_radius / 1e3
        self.ui.depth_slider.setMinimum(min_rad - max_rad)
        self.ui.depth_slider.setMaximum(0)

        self._setup_finite_source()

        self.plot_map()
        self.update()
        self.set_info()

    @QtCore.Slot()
    def on_select_remote_connection_button_released(self):
        text, ok = QtGui.QInputDialog.getText(
            self,
            "Remote Instaseis Connection",
            "Enter URL to remote Instaseis Server:",
        )
        if not ok:
            return
        text = str(text)

        self.instaseis_db = open_db(text)

        # Adjust depth slider to the DB.
        max_rad = self.instaseis_db.info.max_radius / 1e3
        min_rad = self.instaseis_db.info.min_radius / 1e3
        self.ui.depth_slider.setMinimum(min_rad - max_rad)
        self.ui.depth_slider.setMaximum(0)

        self._setup_finite_source()

        self.plot_map()
        self.update()
        self.set_info()

    @QtCore.Slot()
    def on_open_srf_file_button_released(self):
        pwd = os.getcwd()
        self.finite_src_file = str(
            QtGui.QFileDialog.getOpenFileName(
                self,
                "Choose *.srf or *.param File",
                pwd,
                "Standard Rupture Format (*.srf);;"
                "USGS finite source files (*.param)",
            )
        )
        if not self.finite_src_file:
            return
        if self.finite_src_file.endswith(".srf"):
            self.finite_source = FiniteSource.from_srf_file(
                self.finite_src_file, normalize=True
            )
        elif self.finite_src_file.endswith(".param"):
            self.finite_source = FiniteSource.from_usgs_param_file(
                self.finite_src_file
            )
        else:
            raise IOError(
                "unknown file type *.%s" % self.finite_src_file.split(".")[-1]
            )

        self._setup_finite_source()
        self.update()
        self.set_info()

    def _setup_finite_source(self):
        if self.finite_source is None:
            return
        if self.instaseis_db is not None:
            # this is a super uggly construction: if you open a different DB,
            # it will not use the original finite source, but the one already
            # messed up for the previously used DB. Not fixing it here, but in
            # the new GUI we should.

            # self.finite_source.set_sliprate_lp(
            #     dt=self.instaseis_db.info.dt,
            # nsamp=self.instaseis_db.info.npts,
            #     freq=1.0/self.instaseis_db.info.period)

            nsamp = (
                int(self.instaseis_db.info.period / self.finite_source[0].dt)
                * 50
            )
            self.finite_source.resample_sliprate(
                dt=self.finite_source[0].dt, nsamp=nsamp
            )
            self.finite_source.lp_sliprate(
                freq=1.0 / self.instaseis_db.info.period
            )
            self.finite_source.resample_sliprate(
                dt=self.instaseis_db.info.dt, nsamp=self.instaseis_db.info.npts
            )

        self.finite_source.compute_centroid()
        self.plot_mt_finite()
        self.plot_cmt_sliprate()

    def set_info(self):
        info_str = ""
        if self.finite_source is not None:
            info_str += str(self.finite_source) + "\n"
        else:
            info_str += str(self.source) + "\n"
        if self.instaseis_db is not None:
            info_str += str(self.instaseis_db) + "\n"
        self.ui.info_text.setText(info_str)

    @QtCore.Slot()
    def on_load_source_button_released(self):
        pwd = os.getcwd()
        self.source_file = str(
            QtGui.QFileDialog.getOpenFileName(self, "Choose Source File", pwd)
        )
        if not self.source_file:
            return

        s = Source.parse(self.source_file)
        self.ui.m_rr.setValue(s.m_rr)
        self.ui.m_pp.setValue(s.m_pp)
        self.ui.m_rp.setValue(s.m_rp)
        self.ui.m_tt.setValue(s.m_tt)
        self.ui.m_rt.setValue(s.m_rt)
        self.ui.m_tp.setValue(s.m_tp)

        self.ui.source_longitude.setValue(s.longitude)
        self.ui.source_latitude.setValue(s.latitude)
        self.ui.depth_slider.setValue(-s.depth_in_m / 1e3)
        self.set_info()

    @QtCore.Slot("double")
    def on_source_latitude_valueChanged(self, *args):
        self.update()

    @QtCore.Slot("double")
    def on_source_longitude_valueChanged(self, *args):
        self.update()

    @QtCore.Slot("double")
    def on_receiver_latitude_valueChanged(self, *args):
        self.update()

    @QtCore.Slot("double")
    def on_receiver_longitude_valueChanged(self, *args):
        self.update()

    @QtCore.Slot("double")
    def on_m_rr_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @QtCore.Slot("double")
    def on_m_tt_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @QtCore.Slot("double")
    def on_m_pp_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @QtCore.Slot("double")
    def on_m_rt_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @QtCore.Slot("double")
    def on_m_rp_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @QtCore.Slot("double")
    def on_m_tp_valueChanged(self, *args):
        self._draw_mt()
        self.update()

    @property
    def source_depth(self):
        value = int(-1.0 * int(self.ui.depth_slider.value()))
        return value

    @QtCore.Slot()
    def on_depth_slider_valueChanged(self, *args):
        self.ui.depth_label.setText("Depth: %i km" % self.source_depth)
        self.update()

    @QtCore.Slot()
    def on_strike_slider_valueChanged(self, *args):
        self.ui.strike_value.setText("%i" % self.ui.strike_slider.value())
        self._draw_mt()
        self.update()

    @QtCore.Slot()
    def on_dip_slider_valueChanged(self, *args):
        self.ui.dip_value.setText("%i" % self.ui.dip_slider.value())
        self._draw_mt()
        self.update()

    @QtCore.Slot()
    def on_rake_slider_valueChanged(self, *args):
        self.ui.rake_value.setText("%i" % self.ui.rake_slider.value())
        self._draw_mt()
        self.update()

    def autoRange(self):
        widgets = [getattr(self.ui, "%s_graph" % _i) for _i in ("z", "n", "e")]
        widgets.sort(key=lambda x: x.ptp)
        widgets[-1].autoRange()

    @QtCore.Slot()
    def on_reset_view_button_released(self, *args):
        self.autoRange()

    @QtCore.Slot()
    def on_resample_check_box_stateChanged(self):
        resample = bool(self.ui.resample_check_box.checkState())
        self.ui.resample_factor.setEnabled(resample)
        self.ui.sr_ref_label.setEnabled(resample)
        self.update()

    @QtCore.Slot("double")
    def on_resample_factor_valueChanged(self, *args):
        self.update()

    @QtCore.Slot()
    def on_tt_times_stateChanged(self):
        self.update()

    @QtCore.Slot()
    def on_lowpass_check_box_stateChanged(self):
        resample = bool(self.ui.lowpass_check_box.checkState())
        self.ui.lowpass_period.setEnabled(resample)
        self.ui.lowpass_label.setEnabled(resample)
        self.update()

    @QtCore.Slot("double")
    def on_lowpass_period_valueChanged(self, *args):
        self.update()

    @QtCore.Slot()
    def on_highpass_check_box_stateChanged(self):
        resample = bool(self.ui.highpass_check_box.checkState())
        self.ui.highpass_period.setEnabled(resample)
        self.ui.highpass_label.setEnabled(resample)
        self.update()

    @QtCore.Slot("double")
    def on_highpass_period_valueChanged(self, *args):
        self.update()

    @QtCore.Slot()
    def on_zero_phase_check_box_stateChanged(self):
        self.update()

    @QtCore.Slot("int")
    def on_components_combo_currentIndexChanged(self):
        self.update()

    @QtCore.Slot()
    def on_finsource_tab_currentChanged(self):
        self.update()

    @QtCore.Slot()
    def on_source_tab_currentChanged(self):
        self._draw_mt()
        self.update()
        self.autoRange()

    @QtCore.Slot()
    def on_update_button_released(self):
        self.update(force=True)

    @QtCore.Slot()
    def on_load_stations_button_released(self):
        pwd = os.getcwd()
        self.stations_file = str(
            QtGui.QFileDialog.getOpenFileName(
                self, "Choose Stations File", pwd
            )
        )
        if not self.stations_file:
            return

        self.receivers = Receiver.parse(self.stations_file)
        recnames = []
        for _r in self.receivers:
            recnames.append("%s.%s" % (_r.network, _r.station))

        self.ui.stations_combo.clear()
        self.ui.stations_combo.addItems(recnames)

        self._plot_bg_receivers()

    @QtCore.Slot("int")
    def on_stations_combo_currentIndexChanged(self):
        idx = self.ui.stations_combo.currentIndex()
        self.ui.receiver_longitude.setValue(self.receivers[idx].longitude)
        self.ui.receiver_latitude.setValue(self.receivers[idx].latitude)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            if (
                source.parent()
                in [self.ui.z_graph, self.ui.n_graph, self.ui.e_graph]
                and event.buttons() == QtCore.Qt.NoButton
            ):
                try:
                    tt = float(
                        self.ui.z_graph.mapToView(
                            pg.Point(event.pos().x(), event.pos().y())
                        ).x()
                    )
                    closest_phase = min(
                        self.tts, key=lambda x: abs(x.time - tt)
                    )
                    tooltipstr = (
                        "Mouse at %6.2f s, closest phase = %s, "
                        "arriving at %6.2f s"
                        % (tt, closest_phase.name, closest_phase.time)
                    )
                except Exception:
                    tooltipstr = ""

                self.ui.z_graph.setToolTip(tooltipstr)
                self.ui.n_graph.setToolTip(tooltipstr)
                self.ui.e_graph.setToolTip(tooltipstr)

        return QtGui.QMainWindow.eventFilter(self, source, event)


def launch():
    # Automatically compile all ui files if they have been changed.
    compile_and_import_ui_files()

    # Launch and open the window.
    app = QApplication(sys.argv)
    window = Window()

    # Show and bring window to foreground.
    window.show()
    app.installEventFilter(window)
    window.raise_()
    os._exit(app.exec_())
