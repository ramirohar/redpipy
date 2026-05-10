"""
redpipy.osci
~~~~~~~~~~~~

RedPitaya's oscilloscope.


:copyright: 2024 by redpipy Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Generator, Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import common
from .analog import MAXIMUM_SAMPLING_RATE
from .rpwrap import RPBoard, acq, acq_axi, constants


def calculate_best_decimation_axi(trace_duration: float) -> constants.Decimation:
    """Calculate the best decimation in order to acquire a signal
    of a certain duration.

    Parameter
    ---------
    trace_duration
        Duration of the trace in seconds
    """

    for decimation in get_args(common.DECIMATION_VALUES):
        sampling_rate = MAXIMUM_SAMPLING_RATE / decimation
        current_trace_duration = constants.DMA_BUFFER_SIZE / sampling_rate
        if current_trace_duration >= trace_duration:
            return getattr(constants.Decimation, "DEC_{:d}".format(decimation))

    raise ValueError(
        "Could not find suitable decimation value for {trace_duration} seconds."
    )


def calculate_amount_datapoints(min_trace_duration: float, sampling_rate: float) -> int:
    """Calculate the least amount of datapoints that verifies
    trace_duration <= amount_datapoints/sampling_rate.

    Parameter
    ---------
    min_trace_duration
        Duration of the minimum desired trace in seconds.
    sampling_rate
        RP's sampling rate.
    """
    amount_datapoints = math.ceil(min_trace_duration * sampling_rate)
    assert 0 < amount_datapoints <= constants.DMA_BUFFER_SIZE
    return amount_datapoints


class AxiChannel:
    """Oscilloscope channel."""

    channel: constants.Channel

    # Is there a real way to enable/disable a channel?
    enabled: bool = False

    def __init__(self, channel: Literal[1, 2], gain: Literal[1, 5] = 1):
        if channel == 1:
            self.channel = constants.Channel.CH_1
        elif channel == 2:
            self.channel = constants.Channel.CH_2

        self.set_gain(gain)

        self.enabled = False

    def get_trace(
        self,
        delay_samples: int,
        size: int = constants.DMA_BUFFER_SIZE,
        out: npt.NDArray | None = None,
    ) -> npt.NDArray[np.float32]:
        """Get trace (in volts)."""
        # TODO: This works for full buffer, but I think I have to change it
        # when I want less than that
        pointer_at_trig = acq_axi.get_write_pointer_at_trig(self.channel)
        data_pointer = pointer_at_trig + delay_samples
        return acq_axi.get_data_vnp(
            self.channel, data_pointer, size=size, np_buffer=out
        )

    def get_trace_raw(
        self, delay_samples: int, size: int = constants.DMA_BUFFER_SIZE
    ) -> npt.NDArray[np.int16]:
        """Get trace (in ADU)."""
        pointer = acq_axi.get_write_pointer_at_trig(self.channel) + delay_samples
        return acq_axi.get_data_raw(self.channel, pointer, size=size)

    def set_gain(self, gain: Literal[1, 5]):
        if gain == 1:
            acq.set_gain(self.channel, constants.PinState.LOW)
        elif gain == 5:
            acq.set_gain(self.channel, constants.PinState.HIGH)

    def enable(self):
        if not self.enabled:
            acq_axi.enable(self.channel, True)
            self.enabled = True

    def disable(self):
        if self.enabled:
            acq_axi.enable(self.channel, False)
            self.enabled = False


class AxiOscilloscope(RPBoard):
    """Oscilloscope"""

    def __init__(
        self, channel_config: common.ChannelConfig = common.ChannelConfig.CH1_ONLY
    ) -> None:
        super().__init__()
        self._memory_start, self._memory_size = acq_axi.get_memory_region()
        self.configure_memory(channel_config)
        self.channel1 = AxiChannel(1)
        self.channel2 = AxiChannel(2)
        self._channel_config = self.get_channel_config()
        self.configure_trigger()
        self.set_timebase(1)
        self.set_trigger_delay(self.channel1, 1)
        self.set_trigger_delay(self.channel2, 1)
        self._wait_after_trigger = 0

    def configure_memory(self, channel_config: common.ChannelConfig | None = None):
        if channel_config is None:
            channel_config = self.get_channel_config()

        assert (self._memory_size % 16) == 0
        size_samples = self._memory_size // 4
        # The data is saved in 32-bit chunks (4 Bytes per sample)
        if channel_config == common.ChannelConfig.CH1_ONLY:
            acq_axi.set_buffer_samples(
                constants.Channel.CH_1, self._memory_start, size_samples
            )
        elif channel_config == common.ChannelConfig.CH2_ONLY:
            acq_axi.set_buffer_samples(
                constants.Channel.CH_2, self._memory_start, size_samples
            )
        elif channel_config == common.ChannelConfig.BOTH_CH:
            acq_axi.set_buffer_samples(
                constants.Channel.CH_1, self._memory_start, size_samples // 2
            )
            acq_axi.set_buffer_samples(
                constants.Channel.CH_2,
                self._memory_start + self._memory_size // 2,
                size_samples // 2,
            )

    def get_channel_config(self) -> common.ChannelConfig:
        if self.channel1.enabled and self.channel2.enabled:
            channel_config = common.ChannelConfig.BOTH_CH
        elif self.channel1.enabled and not self.channel2.enabled:
            channel_config = common.ChannelConfig.CH1_ONLY
        elif not self.channel1.enabled and self.channel2.enabled:
            channel_config = common.ChannelConfig.CH2_ONLY
        else:
            channel_config = common.ChannelConfig.NO_CHANNELS
        return channel_config

    def get_metadata(self) -> Generator[tuple[Any, Any], Any, None]:
        yield from self.device_metadata.items()
        yield from self.get_timebase_settings().items()
        yield from self.get_trigger_settings().items()

    def get_timebase_settings(self) -> dict[str, Any]:
        """Get timebase settings."""
        trigger_delay_ch1 = acq_axi.get_trigger_delay(self.channel1.channel)
        trigger_delay_ch2 = acq_axi.get_trigger_delay(self.channel2.channel)
        sampling_rate = acq.get_sampling_rate_hz()
        return dict(
            decimation=acq_axi.get_decimation_factor(),
            sampling_rate=sampling_rate,
            trace_duration=self._amount_datapoints / sampling_rate,
            trigger_delay_ch1=trigger_delay_ch1 / sampling_rate,
            trigger_delay_ch1_samples=trigger_delay_ch1,
            trigger_delay_ch2=trigger_delay_ch2 / sampling_rate,
            trigger_delay_ch2_samples=trigger_delay_ch2,
        )

    def get_trigger_settings(self) -> dict[str, Any]:
        """Get trigger settings.

        The trigger source and edge values are read from the Oscilloscope
        class internal configuration. Instead, the trigger level value is
        asked to the hardware.

        """
        if self._trigger_src == constants.AcqTriggerSource.DISABLED:
            return dict(source="disabled", level=None, positive_edge=None)
        elif self._trigger_src == constants.AcqTriggerSource.NOW:
            return dict(source="now", level=None, positive_edge=None)

        source, positive_edge = common.TRIGGER_MAP.inv[self._trigger_src]
        tch = common.TRIGGER_CH_MAP[source]
        return dict(
            source=source,
            level=acq.get_trigger_level(tch),
            positive_edge=positive_edge,
        )

    def get_timevector_raw(
        self, size: int = constants.DMA_BUFFER_SIZE
    ) -> npt.NDArray[np.int64]:
        """Get timevector (in samples)."""
        return (
            np.arange(
                size, dtype=np.int64
            )  # TODO: maybe I have to change this to account for t=0
        )

    def get_timevector(
        self, size: int = constants.DMA_BUFFER_SIZE
    ) -> npt.NDArray[np.float64]:
        """Get timevector (in seconds)."""
        # TODO: update docs to take into account new parameter
        return self.get_timevector_raw(size=size) / acq.get_sampling_rate_hz()

    def get_data(self, raw: bool = False) -> pd.DataFrame:
        """Get data (time, and traces of enabled channels"""
        # TODO: update docs to take into account new parameter
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        timebase_settings = self.get_timebase_settings()
        delay_samples_ch1 = timebase_settings["trigger_delay_ch1_samples"]
        delay_samples_ch2 = timebase_settings["trigger_delay_ch2_samples"]

        if raw:
            out: dict[str, npt.NDArray[Any]] = dict(
                time=self.get_timevector_raw(size=self._amount_datapoints)
            )
            if self.channel1.enabled:
                out["ch1"] = self.channel1.get_trace_raw(
                    delay_samples=delay_samples_ch1, size=self._amount_datapoints
                )
            if self.channel2.enabled:
                out["ch2"] = self.channel2.get_trace_raw(
                    delay_samples=delay_samples_ch2, size=self._amount_datapoints
                )
        else:
            out: dict[str, npt.NDArray[Any]] = dict(
                time=self.get_timevector(size=self._amount_datapoints)
            )
            if self.channel1.enabled:
                out["ch1"] = self.channel1.get_trace(
                    delay_samples=delay_samples_ch1, size=self._amount_datapoints
                )
            if self.channel2.enabled:
                out["ch2"] = self.channel2.get_trace(
                    delay_samples=delay_samples_ch2, size=self._amount_datapoints
                )

        df = pd.DataFrame(out)
        df.attrs["timestamp"] = timestamp
        for k, v in self.get_metadata():
            df.attrs[k] = v

        return df

    def configure_trigger(
        self,
        *,
        source: Literal["ch1", "ch2", "ext"] = "ch1",
        level: float = 0,
        positive_edge: bool = True,
    ):
        """Configure the trigger

        Parameters
        ----------
        source, optional
            Trigger source, by default "ch1"
        level, optional
            Trigger level (in volts), by default 0
        positive_edge, optional
            Triggering occurs in positive edge, by default True
        """

        src = common.TRIGGER_MAP[(source, positive_edge)]
        tch = common.TRIGGER_CH_MAP[source]

        # Store this to be used when arming trigger.
        self._trigger_src = src
        self._trigger_level = (tch, level)

        acq.reset_fpga()
        acq.set_trigger_level(*self._trigger_level)

    def set_timebase(
        self, trace_duration_hint: float, full_buffer: bool = False
    ) -> float:
        """Configure the timebase by providing a trace duration.

        The duration of acquisition window can only take certain discrete
        values. It is guaranteed that the chosen one will be the shortest
        one that can fit the provided trace duration hint. The actual trace
        duration, is returned.

        Parameters
        ----------
        trace_duration_hint
            Duration of the trace to be measured (in seconds).
        full_buffer
            The full RP buffer size is returned, by default False.
        """
        best_decimation = calculate_best_decimation_axi(trace_duration_hint)
        acq_axi.set_decimation_factor(best_decimation)

        sampling_rate = acq.get_sampling_rate_hz()
        if full_buffer:
            self._amount_datapoints = (
                self._memory_size // 2
                if self._channel_config == common.ChannelConfig.BOTH_CH
                else self._memory_size
            )
        else:
            self._amount_datapoints = calculate_amount_datapoints(
                trace_duration_hint, sampling_rate
            )

        trace_duration = self._amount_datapoints / sampling_rate

        return trace_duration

    def set_decimation(self, decimation_exponent: common.DECIMATION_EXPONENTS) -> float:
        """Configure the timebase by providing a decimation.

        Sets the sampling rate of the oscilloscope by providing the decimation.
        The total duration of the trace is determined by the buffer size (2**14)
        over the sampling rate.

        Parameters
        ----------
        decimation_exponent
            Decimation exponent to calculate the decimation of the sampling rate.
        """

        acq_axi.set_decimation_factor(
            common.DECIMATION_MAP[
                get_args(common.DECIMATION_VALUES)[decimation_exponent]
            ]
        )
        sampling_rate = acq.get_sampling_rate_hz()
        self._amount_datapoints = constants.DMA_BUFFER_SIZE
        return self._amount_datapoints / sampling_rate

    def get_voltage_numpy(
        self,
        channel: Literal["ch1", "ch2"],
        out: np.ndarray | None = None,
        delay_samples: int | None = None,
    ) -> np.ndarray:
        if delay_samples is None:
            timebase_settings = self.get_timebase_settings()
            delay_samples = timebase_settings[f"trigger_delay_{channel}_samples"]
        if channel == "ch1":
            voltage = self.channel1.get_trace(
                delay_samples=delay_samples, size=self._amount_datapoints, out=out
            )
        elif channel == "ch2":
            voltage = self.channel2.get_trace(
                delay_samples=delay_samples, size=self._amount_datapoints, out=out
            )
        else:
            raise ValueError(
                f"{channel} is not a valid value, channel must be either 'ch1' or 'ch2'"
            )
        return voltage

    def set_trigger_delay(
        self,
        channel: AxiChannel,
        delay: float,
        units: Literal["second", "trace"] = "trace",
    ):
        """Set the trigger delay.

        The trigger delay is the amount of traces or seconds that are acquired
        after the trigger.

        For example, if the trace_duration is 1s:

        trigger_delay (traces):
        *  0: Trigger and stop acquiring.
        *  1: Trigger and acquire one full trace.
        *  1.5: Trigger, let half a trace pass and acquire a full trace.
        *  2: Trigger, let one trace pass and acquire a full trace.
        And so on.

        trigger_delay (seconds):
        *  0: Trigger and stop acquiring.
        *  1: Trigger and acquire data for 1 second after stopping.
        *  1.5: Trigger and acquire data for 1.5 seconds after stopping.
        *  2: Trigger and acquire data for 2 seconds after stopping.
        And so on.

        Parameters
        ----------
        delay
            Amount of traces (or seconds) during which data is acquired after triggering.
        units, optional
            Units in which the delay is specified, either "trace" or "second" by default trace.
        """

        if units == "second":
            delay_samples = math.ceil(delay * acq.get_sampling_rate_hz())
        elif units == "trace":
            delay_samples = delay * self._amount_datapoints

        acq_axi.set_trigger_delay(channel.channel, int(delay_samples))
        return delay_samples

    def wait_until_done(self, channel: AxiChannel):
        """Wait until the triggering condition has been met."""
        trace_duration = self._amount_datapoints / acq.get_sampling_rate_hz()
        sleep_duration = max(trace_duration / 1000, 100e-6)
        while acq.get_trigger_state() == constants.AcqTriggerState.WAITING:
            time.sleep(sleep_duration)

        while not acq_axi.get_buffer_fill_state(channel.channel):
            time.sleep(sleep_duration)

    def arm_trigger(self, channel: AxiChannel, wait: bool = True) -> None:
        """Arm the trigger.

        If wait is True (default), the thread will lock until the acquisition
        is complete.
        """
        acq.reset_fpga()
        acq.start()
        acq.set_trigger_src(self._trigger_src)
        if wait:
            self.wait_until_done(channel)

    def trigger_now(self, channel: AxiChannel, wait: bool = True):
        """Trigger now.

        If wait is True (default), the thread will lock until the acquisition
        is complete.
        """
        acq.reset_fpga()
        acq.start()
        acq.set_trigger_src(constants.AcqTriggerSource.NOW)
        if wait:
            self.wait_until_done(channel)
