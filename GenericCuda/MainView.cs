using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GenericCuda
{
	public partial class MainView : Form
	{
		// ----- ATTRIBUTES ----- \\
		public string Repopath;


		private int offset = 0;
		private int oldChunkSize = 65536;

		// ----- OBJECTS ----- \\
		public AudioHandling AudioH;
		public CudaHandling CudaH;



		// ----- LAMBDA ----- \\
		public CudaMemoryHandling? MemH => CudaH.MemH;

		public CudaFftHandling? FftH => CudaH.FftH;

		public AudioObject? Track => AudioH.CurrentTrack;

		public long CurrentPointer => Track?.Pointer ?? 0;
		public Image? Wave => AudioH.CurrentWave;



		public bool Initialized => CudaH.Ctx != null && MemH != null;

		public bool Transformed => Track?.Form != 'f';


		// ----- CONSTRUCTOR ----- \\
		public MainView()
		{
			InitializeComponent();

			// Set repopath
			Repopath = GetRepopath(true);

			// Window position
			StartPosition = FormStartPosition.Manual;
			Location = new Point(0, 0);

			// Init. classes
			AudioH = new AudioHandling(listBox_tracks, pictureBox_wave, button_playback, hScrollBar_offset);
			CudaH = new CudaHandling(Repopath, listBox_log, comboBox_devices, label_vram, progressBar_vram);

			// Register events
			hScrollBar_offset.Scroll += (s, e) => offset = hScrollBar_offset.Value;
			listBox_tracks.SelectedIndexChanged += (s, e) => ToggleUI();
			numericUpDown_loggingInterval.ValueChanged += (s, e) => CudaH.LogInterval = (int) numericUpDown_loggingInterval.Value;

			// Start UI
			ToggleUI();
		}





		// ----- METHODS ----- \\
		public string GetRepopath(bool root = false)
		{
			string repo = AppDomain.CurrentDomain.BaseDirectory;

			if (root)
			{
				repo += @"..\..\..\";
			}

			repo = Path.GetFullPath(repo);

			return repo;
		}

		public void ToggleUI()
		{
			// Draw wave
			pictureBox_wave.Image = Wave;

			// Set meta label
			label_meta.Text = Track?.Meta ?? (AudioH.Tracks.Count > 0 ? "No track selected" : "No tracks available");

			// Set offset
			hScrollBar_offset.Value = offset;

			// Set chunk size
			numericUpDown_chunkSize.Value = oldChunkSize;

			// Playback button
			button_playback.Enabled = Track != null && Track.OnHost;

			// Move button
			button_move.Enabled = Track != null && MemH != null;
			button_move.Text = Track != null && Track.OnHost ? "Move to device" : "Move to host";

			// Import button
			button_import.Enabled = Initialized;

			// Button transform
			button_transform.Enabled = Initialized && Track != null;
			button_transform.Text = Transformed ? "I-FFT" : "FFT";

			// Export button
			button_export.Enabled = Track != null && Track.OnHost;

			// Normalize button
			button_normalize.Enabled = Track != null && Track.OnHost;


		}




		// ----- EVENTS ----- \\
		private void numericUpDown_chunkSize_ValueChanged(object sender, EventArgs e)
		{
			// If increased double with limit to maximum
			if (numericUpDown_chunkSize.Value > oldChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Min(numericUpDown_chunkSize.Maximum, oldChunkSize * 2);
			}

			// If decreased half with limit to minimum
			else if (numericUpDown_chunkSize.Value < oldChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Max(numericUpDown_chunkSize.Minimum, oldChunkSize / 2);
			}

			// Set new chunk size
			oldChunkSize = (int) numericUpDown_chunkSize.Value;
		}

		private void button_import_Click(object sender, EventArgs e)
		{
			// OFD for audio files (wav, mp3, flac) at MyMusic
			OpenFileDialog ofd = new OpenFileDialog
			{
				Title = "Import audio file(s)",
				InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic),
				Filter = "Audio files|*.wav;*.mp3;*.flac",
				Multiselect = true,
				CheckFileExists = true
			};

			// OFD show -> AddTrack foreach selected file
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				foreach (string pth in ofd.FileNames)
				{
					AudioH.AddTrack(pth);
				}
			}

			// Select last entry in tracks listbox
			listBox_tracks.SelectedIndex = listBox_tracks.Items.Count - 1;

			ToggleUI();
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			// SFD for audio files (wav) at MyMusic
			SaveFileDialog sfd = new SaveFileDialog
			{
				Title = "Export audio file",
				InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic),
				Filter = "Wav files|*.wav",
				DefaultExt = "wav"
			};

			// SFD show -> ExportTrack
			if (sfd.ShowDialog() == DialogResult.OK)
			{
				Track?.ExportAudioWav(sfd.FileName);

				// MsgBox
				MessageBox.Show("Exported audio file to: \n\n" + sfd.FileName, "Exported", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			ToggleUI();
		}

		private void button_move_Click(object sender, EventArgs e)
		{
			bool chunkFirst = false;
			// If CTRL down: Set internal flag to chunk first
			if (ModifierKeys == Keys.Control)
			{
				chunkFirst = true;
			}

			// Abort if no track selected or no MemH
			if (Track == null || MemH == null)
			{
				return;
			}


			if (chunkFirst)
			{
				// Make chunks of track data
				if (Track.OnHost)
				{
					var chunks = Track.MakeChunks((int) numericUpDown_chunkSize.Value);
					Track.Pointer = MemH.PushData(chunks);
					Track.Data = [];
				}
				else if (Track.OnDevice)
				{
					var chunks = MemH.PullData<float>(Track.Pointer, false);
					Track.AggregateChunks(chunks);
				}

			}
			else
			{
				// Move track data(pointer) beween host and device
				if (Track.OnHost)
				{
					Track.Pointer = MemH.PushData(Track.Data, (int) numericUpDown_chunkSize.Value);
					Track.Data = [];
				}
				else if (Track.OnDevice)
				{
					Track.Data = MemH.PullData<float>(Track.Pointer, true).FirstOrDefault() ?? [];
					Track.Pointer = 0;
				}
			}

			ToggleUI();
		}

		private void button_transform_Click(object sender, EventArgs e)
		{
			// Abort if no track selected or no MemH or no FftH or track on host or no pointer
			if (Track == null || MemH == null || FftH == null || Track.OnHost || CurrentPointer == 0)
			{
				return;
			}

			// Perform FFT or IFFT
			if (Transformed)
			{
				Track.Pointer = FftH.PerformIFFT(CurrentPointer);
				Track.Form = 'f';
			}
			else
			{
				Track.Pointer = FftH.PerformFFT(CurrentPointer);
				Track.Form = 'c';
			}

			ToggleUI();
		}

		private void button_normalize_Click(object sender, EventArgs e)
		{
			Track?.Normalize();

			ToggleUI();
		}
	}
}
