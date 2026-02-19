import os
import glob
import h5py
import numpy as np
import mne

class DataPreprocessing:
    def __init__(self):
        base_path = "F:/EEGDataset/MASS/SS3"
        self.data_files_path = os.path.join(base_path, "data")
        self.anno_files_path = os.path.join(base_path, "annoations")
        self.save_files_path = os.path.join(base_path, "data_anno_savefiles")

        os.makedirs(self.data_files_path, exist_ok=True)
        os.makedirs(self.anno_files_path, exist_ok=True)
        os.makedirs(self.save_files_path, exist_ok=True)

        self.data_files_names = []
        self.anno_files_names = []
        self.save_files_names = []

        self.raw_data = None
        self.raw_annotations = None

        self.segment_seconds = 30
        self.sampling_frequency = 100

        self.labels = []
        self.data = {}
        self.channel_names = []

        self.desired_channel_names = ["EEG C4", "EEG O2", "EEG F4"]

        # Kept for compatibility (not used here, but may be required elsewhere)
        self.data_labels_generator = data_labels_generate.DataLabelsGenerate()

    def save_files(self, index: int) -> None:
        with h5py.File(self.save_files_names[index], "w") as f:
            eeg_group = f.create_group("data")
            for ch in self.channel_names:
                eeg_group.create_dataset(f"{ch}_data", data=self.data[ch])

            labels_group = f.create_group("labels")
            labels_group.create_dataset("stage_labels", data=self.labels)

    def get_files_names(self) -> None:
        self.data_files_names = glob.glob(os.path.join(self.data_files_path, "*PSG.edf"))
        base_names = [os.path.basename(f)[:-8] for f in self.data_files_names]

        self.anno_files_names = []
        self.save_files_names = []

        for bn in base_names:
            anno_pattern = f"{bn} Annotations.edf"
            anno_matches = glob.glob(os.path.join(self.anno_files_path, anno_pattern))
            if not anno_matches:
                continue

            self.anno_files_names.append(anno_matches[0])
            self.save_files_names.append(os.path.join(self.save_files_path, f"{bn}.h5"))

        self.data_files_names = self.data_files_names[: len(self.anno_files_names)]
        self.files_counts = len(self.data_files_names)

    def get_data_labels(self) -> None:
        """Extract labeled EEG segments without artifact removal."""
        self.data.clear()
        self.labels.clear()

        stage_map = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,
            "Sleep stage R": 4,
        }

        sfreq = self.sampling_frequency
        total_len = int(self.raw_data.times[-1] * sfreq)

        label_seq = np.full(total_len, -1, dtype=int)

        for onset, duration, desc in zip(
            self.raw_annotations.onset,
            self.raw_annotations.duration,
            self.raw_annotations.description,
        ):
            start = int(onset * sfreq)
            end = int((onset + duration) * sfreq)
            if desc in stage_map:
                label_seq[start:end] = stage_map[desc]

        valid_idx = np.where(label_seq != -1)[0]
        if len(valid_idx) == 0:
            return

        label_seq = label_seq[valid_idx]

        valid_data = {}
        for ch in self.channel_names:
            full_data = self.raw_data.get_data(picks=ch)[0, :]
            valid_data[ch] = full_data[valid_idx]

        seg_len = int(self.segment_seconds * sfreq)
        total_segments = len(label_seq) // seg_len
        if total_segments == 0:
            return

        label_seq = label_seq[: total_segments * seg_len].reshape(-1, seg_len)

        for ch in self.channel_names:
            valid_data[ch] = valid_data[ch][: total_segments * seg_len].reshape(
                -1, seg_len
            )

        self.labels = []
        for segment_labels in label_seq:
            label = np.bincount(segment_labels).argmax()
            self.labels.append(label)

        for ch in self.channel_names:
            self.data[ch] = valid_data[ch]

        assert (
            len(self.labels) == self.data[self.channel_names[0]].shape[0]
        ), "Mismatch between segments and labels."

    def data_preprocessing(self) -> None:
        self.get_files_names()

        for i, data_file in enumerate(self.data_files_names):
            save_path = self.save_files_names[i]
            if os.path.exists(save_path):
                continue

            fif_name = data_file.replace(".edf", "_raw.fif")
            if not os.path.exists(fif_name):
                self.raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=True)
                self.raw_data.save(fif_name, overwrite=True)
            else:
                self.raw_data = mne.io.read_raw_fif(fif_name, preload=True)

            self.raw_data.resample(self.sampling_frequency)
            self.raw_data.filter(l_freq=0.3, h_freq=40.0, verbose=False)
            self.raw_data.notch_filter(freqs=49, verbose=False)

            self.channel_names = [
                ch
                for ch in self.raw_data.info["ch_names"]
                if any(d in ch for d in self.desired_channel_names)
            ]

            self.raw_annotations = mne.read_annotations(self.anno_files_names[i])

            self.get_data_labels()
            self.save_files(i)


if __name__ == "__main__":
    processor = DataPreprocessing()
    processor.data_preprocessing()
