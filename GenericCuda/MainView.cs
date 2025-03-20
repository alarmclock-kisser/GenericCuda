using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
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

		private static GuiBuilder? Builder;

		// ----- LAMBDA ----- \\
		public CudaMemoryHandling? MemH => CudaH.MemH;
		public CudaFftHandling? FftH => CudaH.FftH;
		public CudaKernelHandling? KernelH => CudaH.KernelH;


		public AudioObject? Track => AudioH.CurrentTrack;
		public string? SelectedKernelEntry => Path.GetFileNameWithoutExtension(KernelH?[listBox_kernels.SelectedIndex]);
		public string? SelectedKernelName => KernelH?.Kernel?.KernelName;


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
			Builder = new GuiBuilder(this);


			// Register events
			hScrollBar_offset.Scroll += (s, e) => offset = hScrollBar_offset.Value;
			listBox_tracks.SelectedIndexChanged += (s, e) => ToggleUI();
			numericUpDown_loggingInterval.ValueChanged += (s, e) => CudaH.LogInterval = (int) numericUpDown_loggingInterval.Value;
			panel_exportLog.MouseMove += (s, e) => ToggleUI();
			button_export.MouseMove += (s, e) => ToggleUI();
			listBox_kernels.SelectedIndexChanged += (s, e) => ToggleUI();
			listBox_kernels.DoubleClick += (s, e) => button_loadKernel.PerformClick();


			// Event handler for CTRL down
			KeyDown += (s, e) =>
			{
				if (e.KeyCode == Keys.ControlKey)
				{
					ToggleUI();
				}
			};


			// Event: KeyDown & MouseLeave for kernel string textbox
			textBox_kernelString.KeyDown += (s, e) =>
			{
				if (e.KeyCode == Keys.Enter || e.KeyCode == Keys.V)
				{
					ToggleUI();
				}
			};
			textBox_kernelString.MouseLeave += (s, e) => ToggleUI();



			// Start UI
			Builder.BuildParameters(true);
			CudaH.FillKernelsListbox(listBox_kernels);
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

			// Set kernel loaded label
			label_kernelLoaded.Text = KernelH?.Kernel != null ? KernelH.Kernel.KernelName : "No kernel loaded";

			// Set offset
			hScrollBar_offset.Value = offset;

			// Set chunk size
			numericUpDown_chunkSize.Value = oldChunkSize;

			// Add vertical scrollbar if lines in textbox > 25 (always horizontal)
			textBox_kernelString.ScrollBars = textBox_kernelString.Lines.Length > 25 ? ScrollBars.Both : ScrollBars.Horizontal;

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
			button_export.Enabled = ModifierKeys == Keys.Control || Track != null && Track.OnHost;
			button_export.Text = ModifierKeys == Keys.Control ? "Export log" : "Export";

			// Normalize button
			button_normalize.Enabled = Track != null && Track.OnHost;

			// Compile button
			button_compile.Enabled = Initialized && KernelH?.PrecompileKernelString(textBox_kernelString.Text, true) != null;

			// Load kernel button
			button_loadKernel.Enabled = Initialized && listBox_kernels.SelectedIndex >= 0;

			// Execute kernel button
			button_executeKernel.Enabled = Initialized && Track != null && MemH != null && KernelH != null && KernelH.Kernel != null && CurrentPointer != 0 || SelectedKernelName == null;

		}

		public string ExportLog()
		{
			// SFD for log file at MyDocuments
			SaveFileDialog sfd = new()
			{
				Title = "Export log file",
				InitialDirectory = Path.Combine(Repopath, "Resources\\Logs"),
				Filter = "Text files|*.txt",
				FileName = "log_" + DateTime.Now.ToString("yyyy-MM-dd_HH-mm"),
				DefaultExt = "txt"
			};

			// SFD show -> Export log
			if (sfd.ShowDialog() == DialogResult.OK)
			{
				// Aggregate every log line
				string logText = "~~~~~~~ LOG from " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " ~~~~~~~\n\n";
				foreach (string line in listBox_log.Items)
				{
					logText += line + "\n";
				}

				File.WriteAllText(sfd.FileName, logText);

				// MsgBox
				MessageBox.Show("Exported log file to: \n\n" + sfd.FileName, "Exported", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			return Path.GetFullPath(sfd.FileName);
		}


		// ----- EVENTS ----- \\
		// Toggles
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

		private void checkBox_overwrite_CheckedChanged(object sender, EventArgs e)
		{
			// Toggle color to red if checked, else to black
			checkBox_overwrite.ForeColor = checkBox_overwrite.Checked ? Color.Red : Color.Black;
			checkBox_overwrite.Text = checkBox_overwrite.Checked ? "Overwrite (!)" : "Overwrite?";
		}


		// I/O
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
			// If CTRL down: Export log
			if (ModifierKeys == Keys.Control)
			{
				ExportLog();
				return;
			}

			// Abort if no Track or data OnHost
			if (Track == null || !Track.OnHost)
			{
				// MsgBox
				MessageBox.Show("No track selected or track not on Host", "Export failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// SFD for audio files (wav) at MyMusic
			SaveFileDialog sfd = new()
			{
				Title = "Export audio file",
				InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic),
				Filter = "Wav files|*.wav",
				FileName = Track?.Name ?? "track_" + Track?.GetHashCode(),
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


		// Host operations
		private void button_normalize_Click(object sender, EventArgs e)
		{
			// Abort if no track selected or track on device
			if (Track == null || Track.OnDevice)
			{
				// MsgBox
				MessageBox.Show("No track selected or track not on Host", "Normalize failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			Track?.Normalize();

			ToggleUI();
		}


		// CUDA move
		private void button_move_Click(object sender, EventArgs e)
		{
			// Get flag? if CTRL is down
			bool chunkFirst = false;
			if (ModifierKeys == Keys.Control)
			{
				chunkFirst = true;
			}

			// Abort if no track selected or no MemH
			if (Track == null || MemH == null)
			{
				// MsgBox
				MessageBox.Show("No track selected or no memory handling available", "Move failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
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


		// CUDA transform
		private void button_transform_Click(object sender, EventArgs e)
		{
			// Abort if no track selected or no MemH or no FftH or track on host or no pointer
			if (Track == null || MemH == null || FftH == null || Track.OnHost || CurrentPointer == 0)
			{
				// MsgBox
				MessageBox.Show("No track selected or track not on Device", "Transform failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
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


		// CUDA kernel 
		private void button_compile_Click(object sender, EventArgs e)
		{
			// Check KernelH
			if (KernelH == null)
			{
				// MsgBox
				MessageBox.Show("No kernel handling available", "Kernel failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			string? kernelName = KernelH.PrecompileKernelString(textBox_kernelString.Text, false);

			// Test precompile kernel string
			if (kernelName == null)
			{
				// MsgBox
				MessageBox.Show("Failed to pre-compile kernel string! \n\nForgot something?", "Kernel failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Abort if overwrite? and file already exists
			string ptxPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX", kernelName + "Kernel.ptx");
			if (File.Exists(ptxPath) && !checkBox_overwrite.Checked)
			{
				// MsgBox
				if (MessageBox.Show("Kernel file already exists! \n\nOverwrite?", "Kernel failed", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.No)
				{
					KernelH.Log("Kernel file already exists, aborting", "No overwrite", 1);
					return;
				}
				else
				{
					File.Delete(ptxPath);
				}
			}

			// Compile kernel string
			ptxPath = KernelH.CompileString(textBox_kernelString.Text);

			// Load kernel
			KernelH.LoadKernel(ptxPath);

			// Fill kernels listbox
			CudaH.FillKernelsListbox(listBox_kernels);

			ToggleUI();
		}

		private void button_loadKernel_Click(object sender, EventArgs e)
		{
			// Load selected Kernel from listbox kernels
			if (KernelH == null || SelectedKernelEntry == null || listBox_kernels.SelectedIndex < 0)
			{
				// MsgBox
				MessageBox.Show("No kernel handling available or no kernel selected", "Kernel failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			KernelH.LoadKernelByName(SelectedKernelEntry.Replace("Kernel", "") ?? "");

			// Set kernel string to textbox if empty and kernel loaded
			if (textBox_kernelString.Text.Trim() == string.Empty && KernelH?.Kernel != null)
			{
				textBox_kernelString.Text = KernelH.KernelString ?? "";
			}

			Builder.BuildParameters(false);
			ToggleUI();
		}

		private void button_executeKernel_Click(object sender, EventArgs e)
		{
			// Abort if no track selected or no MemH or no KernelH or no pointer
			if (Track == null || MemH == null || KernelH == null || CurrentPointer == 0 || KernelH.Kernel == null)
			{
				// MsgBox
				MessageBox.Show("No track selected or no kernel available", "Kernel failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get all controls with "numericUpDown_param" in name with their number ordered
			NumericUpDown[] paramControls = panel_kernelParams.Controls.OfType<NumericUpDown>().Where(c => c.Name.Contains("numericUpDown_param")).OrderBy(c => int.Parse(c.Name.Replace("numericUpDown_param", ""))).ToArray();

			// DEBUG log
			KernelH.Log("Found " + paramControls.Length + " parameters for kernel execution", "", 2);

			// Get parameter values as object array
			object[] paramValues = paramControls.Select(c => (object) c.Value).ToArray();

			// DEBUG log
			string paramValuesString = string.Join(", ", paramValues.Select(p => p.ToString()));
			KernelH.Log("Parameter values: " + paramValuesString, "", 2);

			// Execute kernel with parameters
			KernelH.ExecuteKernel(CurrentPointer, paramValues);

			ToggleUI();
		}

		

		
	}
}
