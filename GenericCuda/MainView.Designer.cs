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
			button_normalize = new Button();
			label_logging = new Label();
			numericUpDown_loggingInterval = new NumericUpDown();
			button_transform = new Button();
			numericUpDown_chunkSize = new NumericUpDown();
			button_move = new Button();
			button_import = new Button();
			button_export = new Button();
			textBox_time = new TextBox();
			comboBox_devices = new ComboBox();
			label_vram = new Label();
			progressBar_vram = new ProgressBar();
			((System.ComponentModel.ISupportInitialize) pictureBox_wave).BeginInit();
			groupBox_controls.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_loggingInterval).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).BeginInit();
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
			groupBox_controls.Controls.Add(button_normalize);
			groupBox_controls.Controls.Add(label_logging);
			groupBox_controls.Controls.Add(numericUpDown_loggingInterval);
			groupBox_controls.Controls.Add(button_transform);
			groupBox_controls.Controls.Add(numericUpDown_chunkSize);
			groupBox_controls.Controls.Add(button_move);
			groupBox_controls.Controls.Add(button_import);
			groupBox_controls.Controls.Add(button_export);
			groupBox_controls.Controls.Add(textBox_time);
			groupBox_controls.Controls.Add(button_playback);
			groupBox_controls.Location = new Point(592, 660);
			groupBox_controls.Name = "groupBox_controls";
			groupBox_controls.Size = new Size(240, 159);
			groupBox_controls.TabIndex = 6;
			groupBox_controls.TabStop = false;
			groupBox_controls.Text = "Controls";
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
			// button_export
			// 
			button_export.Location = new Point(159, 130);
			button_export.Name = "button_export";
			button_export.Size = new Size(75, 23);
			button_export.TabIndex = 7;
			button_export.Text = "Export";
			button_export.UseVisualStyleBackColor = true;
			button_export.Click += button_export_Click;
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
			// MainView
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(844, 991);
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
			((System.ComponentModel.ISupportInitialize) numericUpDown_loggingInterval).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).EndInit();
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
	}
}