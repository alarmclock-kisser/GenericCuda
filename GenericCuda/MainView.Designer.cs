namespace GenericCuda
{
	partial class MainView
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			listBox_log = new ListBox();
			listBox_tracks = new ListBox();
			hScrollBar_offset = new HScrollBar();
			label_meta = new Label();
			pictureBox_wave = new PictureBox();
			button_playback = new Button();
			groupBox_controls = new GroupBox();
			panel_exportLog = new Panel();
			button_export = new Button();
			button_normalize = new Button();
			label_logging = new Label();
			numericUpDown_loggingInterval = new NumericUpDown();
			button_transform = new Button();
			numericUpDown_chunkSize = new NumericUpDown();
			button_move = new Button();
			button_import = new Button();
			textBox_time = new TextBox();
			comboBox_devices = new ComboBox();
			label_vram = new Label();
			progressBar_vram = new ProgressBar();
			textBox_kernelString = new TextBox();
			button_compile = new Button();
			checkBox_overwrite = new CheckBox();
			listBox_kernels = new ListBox();
			button_loadKernel = new Button();
			button_executeKernel = new Button();
			numericUpDown_param1 = new NumericUpDown();
			label_param1 = new Label();
			label_param2 = new Label();
			numericUpDown_param2 = new NumericUpDown();
			panel_param2 = new Panel();
			label_toggleParam2 = new Label();
			label_kernelLoaded = new Label();
			panel_kernelParams = new Panel();
			((System.ComponentModel.ISupportInitialize) pictureBox_wave).BeginInit();
			groupBox_controls.SuspendLayout();
			panel_exportLog.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_loggingInterval).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param1).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param2).BeginInit();
			panel_param2.SuspendLayout();
			SuspendLayout();
			// 
			// listBox_log
			// 
			listBox_log.FormattingEnabled = true;
			listBox_log.ItemHeight = 15;
			listBox_log.Location = new Point(12, 840);
			listBox_log.Name = "listBox_log";
			listBox_log.Size = new Size(574, 139);
			listBox_log.TabIndex = 0;
			// 
			// listBox_tracks
			// 
			listBox_tracks.FormattingEnabled = true;
			listBox_tracks.ItemHeight = 15;
			listBox_tracks.Location = new Point(592, 840);
			listBox_tracks.Name = "listBox_tracks";
			listBox_tracks.Size = new Size(240, 139);
			listBox_tracks.TabIndex = 1;
			// 
			// hScrollBar_offset
			// 
			hScrollBar_offset.Location = new Point(12, 823);
			hScrollBar_offset.Name = "hScrollBar_offset";
			hScrollBar_offset.Size = new Size(574, 14);
			hScrollBar_offset.TabIndex = 2;
			// 
			// label_meta
			// 
			label_meta.AutoSize = true;
			label_meta.Location = new Point(592, 822);
			label_meta.Name = "label_meta";
			label_meta.Size = new Size(89, 15);
			label_meta.TabIndex = 3;
			label_meta.Text = "track meta data";
			// 
			// pictureBox_wave
			// 
			pictureBox_wave.BackColor = Color.White;
			pictureBox_wave.Location = new Point(12, 660);
			pictureBox_wave.Name = "pictureBox_wave";
			pictureBox_wave.Size = new Size(574, 160);
			pictureBox_wave.TabIndex = 4;
			pictureBox_wave.TabStop = false;
			// 
			// button_playback
			// 
			button_playback.Location = new Point(6, 130);
			button_playback.Name = "button_playback";
			button_playback.Size = new Size(23, 23);
			button_playback.TabIndex = 5;
			button_playback.Text = ">";
			button_playback.UseVisualStyleBackColor = true;
			// 
			// groupBox_controls
			// 
			groupBox_controls.Controls.Add(panel_exportLog);
			groupBox_controls.Controls.Add(button_normalize);
			groupBox_controls.Controls.Add(label_logging);
			groupBox_controls.Controls.Add(numericUpDown_loggingInterval);
			groupBox_controls.Controls.Add(button_transform);
			groupBox_controls.Controls.Add(numericUpDown_chunkSize);
			groupBox_controls.Controls.Add(button_move);
			groupBox_controls.Controls.Add(button_import);
			groupBox_controls.Controls.Add(textBox_time);
			groupBox_controls.Controls.Add(button_playback);
			groupBox_controls.Location = new Point(592, 660);
			groupBox_controls.Name = "groupBox_controls";
			groupBox_controls.Size = new Size(240, 159);
			groupBox_controls.TabIndex = 6;
			groupBox_controls.TabStop = false;
			groupBox_controls.Text = "Controls";
			// 
			// panel_exportLog
			// 
			panel_exportLog.Controls.Add(button_export);
			panel_exportLog.Location = new Point(159, 126);
			panel_exportLog.Name = "panel_exportLog";
			panel_exportLog.Size = new Size(75, 23);
			panel_exportLog.TabIndex = 22;
			// 
			// button_export
			// 
			button_export.Location = new Point(0, 0);
			button_export.Name = "button_export";
			button_export.Size = new Size(75, 23);
			button_export.TabIndex = 7;
			button_export.Text = "Export";
			button_export.UseVisualStyleBackColor = true;
			button_export.Click += button_export_Click;
			// 
			// button_normalize
			// 
			button_normalize.Location = new Point(6, 80);
			button_normalize.Name = "button_normalize";
			button_normalize.Size = new Size(75, 23);
			button_normalize.TabIndex = 10;
			button_normalize.Text = "Normalize";
			button_normalize.UseVisualStyleBackColor = true;
			button_normalize.Click += button_normalize_Click;
			// 
			// label_logging
			// 
			label_logging.AutoSize = true;
			label_logging.Location = new Point(159, 54);
			label_logging.Name = "label_logging";
			label_logging.Size = new Size(51, 15);
			label_logging.TabIndex = 10;
			label_logging.Text = "Logging";
			// 
			// numericUpDown_loggingInterval
			// 
			numericUpDown_loggingInterval.Increment = new decimal(new int[] { 10, 0, 0, 0 });
			numericUpDown_loggingInterval.Location = new Point(159, 72);
			numericUpDown_loggingInterval.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
			numericUpDown_loggingInterval.Minimum = new decimal(new int[] { 10, 0, 0, 0 });
			numericUpDown_loggingInterval.Name = "numericUpDown_loggingInterval";
			numericUpDown_loggingInterval.Size = new Size(75, 23);
			numericUpDown_loggingInterval.TabIndex = 10;
			numericUpDown_loggingInterval.Value = new decimal(new int[] { 100, 0, 0, 0 });
			// 
			// button_transform
			// 
			button_transform.Location = new Point(6, 51);
			button_transform.Name = "button_transform";
			button_transform.Size = new Size(75, 23);
			button_transform.TabIndex = 10;
			button_transform.Text = "Transform";
			button_transform.UseVisualStyleBackColor = true;
			button_transform.Click += button_transform_Click;
			// 
			// numericUpDown_chunkSize
			// 
			numericUpDown_chunkSize.Location = new Point(87, 22);
			numericUpDown_chunkSize.Maximum = new decimal(new int[] { 1048576, 0, 0, 0 });
			numericUpDown_chunkSize.Minimum = new decimal(new int[] { 512, 0, 0, 0 });
			numericUpDown_chunkSize.Name = "numericUpDown_chunkSize";
			numericUpDown_chunkSize.Size = new Size(147, 23);
			numericUpDown_chunkSize.TabIndex = 9;
			numericUpDown_chunkSize.Value = new decimal(new int[] { 65536, 0, 0, 0 });
			numericUpDown_chunkSize.ValueChanged += numericUpDown_chunkSize_ValueChanged;
			// 
			// button_move
			// 
			button_move.Location = new Point(6, 22);
			button_move.Name = "button_move";
			button_move.Size = new Size(75, 23);
			button_move.TabIndex = 7;
			button_move.Text = "Move";
			button_move.UseVisualStyleBackColor = true;
			button_move.Click += button_move_Click;
			// 
			// button_import
			// 
			button_import.Location = new Point(159, 101);
			button_import.Name = "button_import";
			button_import.Size = new Size(75, 23);
			button_import.TabIndex = 8;
			button_import.Text = "Import";
			button_import.UseVisualStyleBackColor = true;
			button_import.Click += button_import_Click;
			// 
			// textBox_time
			// 
			textBox_time.Location = new Point(35, 130);
			textBox_time.Name = "textBox_time";
			textBox_time.ReadOnly = true;
			textBox_time.Size = new Size(70, 23);
			textBox_time.TabIndex = 6;
			// 
			// comboBox_devices
			// 
			comboBox_devices.FormattingEnabled = true;
			comboBox_devices.Location = new Point(12, 12);
			comboBox_devices.Name = "comboBox_devices";
			comboBox_devices.Size = new Size(240, 23);
			comboBox_devices.TabIndex = 7;
			comboBox_devices.Text = "Select a CUDA device to initialize";
			// 
			// label_vram
			// 
			label_vram.AutoSize = true;
			label_vram.Location = new Point(12, 38);
			label_vram.Name = "label_vram";
			label_vram.Size = new Size(90, 15);
			label_vram.TabIndex = 8;
			label_vram.Text = "VRAM: 0 / 0 MB\r\n";
			// 
			// progressBar_vram
			// 
			progressBar_vram.Location = new Point(12, 56);
			progressBar_vram.Name = "progressBar_vram";
			progressBar_vram.Size = new Size(240, 12);
			progressBar_vram.TabIndex = 9;
			// 
			// textBox_kernelString
			// 
			textBox_kernelString.AcceptsReturn = true;
			textBox_kernelString.AcceptsTab = true;
			textBox_kernelString.Location = new Point(12, 157);
			textBox_kernelString.MaxLength = 9999999;
			textBox_kernelString.Multiline = true;
			textBox_kernelString.Name = "textBox_kernelString";
			textBox_kernelString.PlaceholderText = "Your kernel string here";
			textBox_kernelString.ScrollBars = ScrollBars.Horizontal;
			textBox_kernelString.Size = new Size(574, 468);
			textBox_kernelString.TabIndex = 10;
			textBox_kernelString.WordWrap = false;
			// 
			// button_compile
			// 
			button_compile.Location = new Point(511, 631);
			button_compile.Name = "button_compile";
			button_compile.Size = new Size(75, 23);
			button_compile.TabIndex = 11;
			button_compile.Text = "Compile";
			button_compile.UseVisualStyleBackColor = true;
			button_compile.Click += button_compile_Click;
			// 
			// checkBox_overwrite
			// 
			checkBox_overwrite.AutoSize = true;
			checkBox_overwrite.Location = new Point(411, 634);
			checkBox_overwrite.Name = "checkBox_overwrite";
			checkBox_overwrite.Size = new Size(82, 19);
			checkBox_overwrite.TabIndex = 12;
			checkBox_overwrite.Text = "Overwrite?";
			checkBox_overwrite.TextAlign = ContentAlignment.MiddleCenter;
			checkBox_overwrite.UseVisualStyleBackColor = true;
			checkBox_overwrite.CheckedChanged += checkBox_overwrite_CheckedChanged;
			// 
			// listBox_kernels
			// 
			listBox_kernels.FormattingEnabled = true;
			listBox_kernels.ItemHeight = 15;
			listBox_kernels.Location = new Point(592, 157);
			listBox_kernels.Name = "listBox_kernels";
			listBox_kernels.Size = new Size(240, 169);
			listBox_kernels.TabIndex = 13;
			// 
			// button_loadKernel
			// 
			button_loadKernel.Location = new Point(592, 332);
			button_loadKernel.Name = "button_loadKernel";
			button_loadKernel.Size = new Size(55, 23);
			button_loadKernel.TabIndex = 14;
			button_loadKernel.Text = "Load";
			button_loadKernel.UseVisualStyleBackColor = true;
			button_loadKernel.Click += button_loadKernel_Click;
			// 
			// button_executeKernel
			// 
			button_executeKernel.BackColor = SystemColors.ActiveBorder;
			button_executeKernel.Location = new Point(777, 332);
			button_executeKernel.Name = "button_executeKernel";
			button_executeKernel.Size = new Size(55, 23);
			button_executeKernel.TabIndex = 15;
			button_executeKernel.Text = "Execute";
			button_executeKernel.UseVisualStyleBackColor = false;
			button_executeKernel.Click += button_executeKernel_Click;
			// 
			// numericUpDown_param1
			// 
			numericUpDown_param1.DecimalPlaces = 12;
			numericUpDown_param1.Increment = new decimal(new int[] { 5, 0, 0, 262144 });
			numericUpDown_param1.Location = new Point(592, 12);
			numericUpDown_param1.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
			numericUpDown_param1.Minimum = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_param1.Name = "numericUpDown_param1";
			numericUpDown_param1.Size = new Size(118, 23);
			numericUpDown_param1.TabIndex = 16;
			numericUpDown_param1.Value = new decimal(new int[] { 1, 0, 0, 0 });
			// 
			// label_param1
			// 
			label_param1.AutoSize = true;
			label_param1.Location = new Point(716, 14);
			label_param1.Name = "label_param1";
			label_param1.Size = new Size(47, 15);
			label_param1.TabIndex = 17;
			label_param1.Text = "Param1";
			// 
			// label_param2
			// 
			label_param2.AutoSize = true;
			label_param2.Location = new Point(716, 43);
			label_param2.Name = "label_param2";
			label_param2.Size = new Size(47, 15);
			label_param2.TabIndex = 19;
			label_param2.Text = "Param2";
			// 
			// numericUpDown_param2
			// 
			numericUpDown_param2.DecimalPlaces = 2;
			numericUpDown_param2.Enabled = false;
			numericUpDown_param2.Location = new Point(0, 0);
			numericUpDown_param2.Maximum = new decimal(new int[] { 99999, 0, 0, 0 });
			numericUpDown_param2.Name = "numericUpDown_param2";
			numericUpDown_param2.Size = new Size(118, 23);
			numericUpDown_param2.TabIndex = 18;
			// 
			// panel_param2
			// 
			panel_param2.Controls.Add(numericUpDown_param2);
			panel_param2.Location = new Point(592, 41);
			panel_param2.Name = "panel_param2";
			panel_param2.Size = new Size(118, 23);
			panel_param2.TabIndex = 20;
			// 
			// label_toggleParam2
			// 
			label_toggleParam2.AutoSize = true;
			label_toggleParam2.Font = new Font("Segoe UI", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			label_toggleParam2.ForeColor = Color.Black;
			label_toggleParam2.Location = new Point(592, 67);
			label_toggleParam2.Name = "label_toggleParam2";
			label_toggleParam2.Size = new Size(109, 13);
			label_toggleParam2.TabIndex = 21;
			label_toggleParam2.Text = "CTRL-click to enable";
			// 
			// label_kernelLoaded
			// 
			label_kernelLoaded.AutoSize = true;
			label_kernelLoaded.Location = new Point(653, 336);
			label_kernelLoaded.Name = "label_kernelLoaded";
			label_kernelLoaded.Size = new Size(97, 15);
			label_kernelLoaded.TabIndex = 22;
			label_kernelLoaded.Text = "No kernel loaded";
			// 
			// panel_kernelParams
			// 
			panel_kernelParams.BackColor = Color.White;
			panel_kernelParams.Location = new Point(592, 361);
			panel_kernelParams.Name = "panel_kernelParams";
			panel_kernelParams.Size = new Size(240, 77);
			panel_kernelParams.TabIndex = 23;
			// 
			// MainView
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			BackColor = SystemColors.ButtonFace;
			ClientSize = new Size(844, 991);
			Controls.Add(panel_kernelParams);
			Controls.Add(label_kernelLoaded);
			Controls.Add(label_toggleParam2);
			Controls.Add(panel_param2);
			Controls.Add(label_param2);
			Controls.Add(label_param1);
			Controls.Add(numericUpDown_param1);
			Controls.Add(button_executeKernel);
			Controls.Add(button_loadKernel);
			Controls.Add(listBox_kernels);
			Controls.Add(checkBox_overwrite);
			Controls.Add(button_compile);
			Controls.Add(textBox_kernelString);
			Controls.Add(progressBar_vram);
			Controls.Add(label_vram);
			Controls.Add(comboBox_devices);
			Controls.Add(groupBox_controls);
			Controls.Add(pictureBox_wave);
			Controls.Add(label_meta);
			Controls.Add(hScrollBar_offset);
			Controls.Add(listBox_tracks);
			Controls.Add(listBox_log);
			MaximizeBox = false;
			MinimumSize = new Size(860, 1030);
			Name = "MainView";
			Text = "MainView";
			((System.ComponentModel.ISupportInitialize) pictureBox_wave).EndInit();
			groupBox_controls.ResumeLayout(false);
			groupBox_controls.PerformLayout();
			panel_exportLog.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize) numericUpDown_loggingInterval).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param1).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_param2).EndInit();
			panel_param2.ResumeLayout(false);
			ResumeLayout(false);
			PerformLayout();
		}

		#endregion

		private ListBox listBox_log;
		private ListBox listBox_tracks;
		private HScrollBar hScrollBar_offset;
		private Label label_meta;
		private PictureBox pictureBox_wave;
		private Button button_playback;
		private GroupBox groupBox_controls;
		private NumericUpDown numericUpDown_chunkSize;
		private Button button_move;
		private Button button_import;
		private Button button_export;
		private TextBox textBox_time;
		private ComboBox comboBox_devices;
		private Label label_vram;
		private ProgressBar progressBar_vram;
		private Button button_transform;
		private Label label_logging;
		private NumericUpDown numericUpDown_loggingInterval;
		private Button button_normalize;
		private TextBox textBox_kernelString;
		private Button button_compile;
		private CheckBox checkBox_overwrite;
		private ListBox listBox_kernels;
		private Button button_loadKernel;
		private Button button_executeKernel;
		private NumericUpDown numericUpDown_param1;
		private Label label_param1;
		private Label label_param2;
		private NumericUpDown numericUpDown_param2;
		private Panel panel_param2;
		private Label label_toggleParam2;
		private Panel panel_exportLog;
		private Label label_kernelLoaded;
		private Panel panel_kernelParams;
	}
}