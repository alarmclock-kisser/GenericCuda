namespace GenericCuda
{
	public class GuiBuilder
	{
		// ----- ATTRIBUTES ----- \\
		private MainView Win;


		public const int WidthParamNumUD = 150;
		public const int DecimalsParamNumUD = 12;

		private int spaceWidth => TextRenderer.MeasureText(" ", ParamsPanel?.Font).Width;
		

		// ----- CONTROLS ----- \\
		private Panel? ParamsPanel => Win.Controls.Find("panel_kernelParams", true).FirstOrDefault() as Panel ?? null;
		private ListBox LogBox => Win.Controls.Find("listBox_log", true).FirstOrDefault() as ListBox ?? new ListBox();

		private List<Label> ParamLabels = [];
		private List<NumericUpDown> ParamNums = [];


		// ----- LAMBDA ----- \\
		private Dictionary<string, Type>? Parameters => Win.KernelH?.KernelParameters ?? null;
		

		// ----- CONSTRUCTOR ----- \\
		public GuiBuilder(MainView win)
		{
			// Set attributes
			this.Win = win;

			// 


		}


		// ----- METHODS ----- \\
		// Log
		public void Log(string message, string inner = "", int layer = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] ";
			msg += "<GUI>";

			for (int i = 0; i <= layer; i++)
			{
				msg += " - ";
			}

			msg += message;

			if (inner != "")
			{
				msg += "  (" + inner + ")";
			}

			if (update)
			{
				LogBox.Items[^1] = msg;
			}
			else
			{
				LogBox.Items.Add(msg);
				LogBox.SelectedIndex = LogBox.Items.Count - 1;
			}
		}


		// BuildParameters
		public void BuildParameters(bool silent = false)
		{
			// Remove every Control from the panel
			ParamsPanel?.Controls.Clear();

			// Get count of params
			int count = Parameters?.Count ?? 0;

			// Abort if panel is null
			if (ParamsPanel == null)
			{
				// Log
				if (!silent)
				{
					Log("No parameters panel found.", "Panel is null", 1);
				}
				return;
			}

			// Abort if count is 0
			if (count == 0 || Parameters == null)
			{
				ParamsPanel.Enabled = false;
				ParamsPanel.Visible = false;

				// Log
				if (!silent)
				{
					Log("No parameters found for kernel.", "Panel is null-sized", 1);
				}
				return;
			}

			// Copy parameters to local dictionary
			Dictionary<string, Type> parameters = new(Parameters);

			// Set panel size (height is 29 per param) + 6 for padding
			ParamsPanel.Height = 29 * count + 6;

			// Skip (remove) first int32 type parameter
			parameters.Remove(Parameters.First(x => x.Value.Name == "Int32").Key);
			if (!silent)
			{
				Log("Removed first Int32 parameter", "First int parameter is always the input pointer", 2);
			}

			// Log
			if (!silent)
			{
				Log("Building parameters for kernel.", "Count: " + count + " , Panel height: " + ParamsPanel.Height, 1);
			}

			// Generate controls for each parameter (Label(type : name) + NumericUpDown(value, width until end of Panel, consicer padding / margin))
			int i = 0;
			foreach (var param in parameters)
			{
				// Skip (continue) if name is caps
				if (param.Key == param.Key.ToUpper())
				{
					// Remove entry from dictionary
					parameters.Remove(Parameters.First(x => x.Key == param.Key).Key);

					// Log
					if (!silent)
					{
						Log("Found param for input pointer", "Name is in caps: " + param.Key, 2);
					}
					continue;
				}

				// Get primitive type
				string type = param.Value.Name;

				// Adjust decimals if comma-separated type
				int decimals = type == "Single" || type == "Double" ? DecimalsParamNumUD : 0;

				// Adjust min / max & increment
				float min = type == "Single" || type == "Double" ? 0.001f : 0;
				float max = type == "Single" || type == "Double" ? 9.999f : 255;
				float inc = type == "Single" || type == "Double" ? 0.0005f : 1;
				float value = type == "Single" || type == "Double" ? 1.0f : 0;

				// Create Label
				Label label = new()
				{
					Name = "label_param" + i,
					Text = "'" + param.Key + "'",
					Location = new Point(3, 3 + 29 * i),
					Width = 80,
					TextAlign = ContentAlignment.MiddleLeft,
					Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right,
				};

				// Create ToolTip with param name (shows after 100ms)
				ToolTip tip = new()
				{
					InitialDelay = 100,
					AutoPopDelay = 5000,
					ReshowDelay = 200,
					ShowAlways = true,
				};

				// Set ToolTip for Label
				tip.SetToolTip(label, type);

				// Create NumericUpDown
				NumericUpDown num = new()
				{
					Name = "numericUpDown_param" + i,
					Location = new Point(ParamsPanel.Width - WidthParamNumUD - 3, 3 + 29 * i),
					Width = WidthParamNumUD,
					Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right,
					Minimum = (decimal) min,
					Maximum = (decimal) max,
					DecimalPlaces = decimals,
					Value = (decimal) value,
					Increment = (decimal) inc
				};

				// Add to panel
				ParamsPanel.Controls.Add(label);
				ParamsPanel.Controls.Add(num);
				
				// Add to lists
				ParamLabels.Add(label);
				ParamNums.Add(num);
				
				// Increment counter
				i++;
			}

			// Enable & show panel
			ParamsPanel.Enabled = true;
			ParamsPanel.Visible = true;
		}


		// GetStringWidth
		public int GetStringWidth(string text = "", Control? control = null)
		{
			// If control is null, return default TextRenderer result
			if (control == null)
			{
				return TextRenderer.MeasureText(text, new Font("Segoe UI", 9)).Width;
			}

			// Get all control properties
			var props = control.GetType().GetProperties();

			// Get control's attributes which return a string
			var stringAttributes = props.Where(p => p.PropertyType == typeof(string));

			// Try all attributes for non-empty string
			int i = 0;
			while (text == "" && i < stringAttributes.Count())
			{
				text = stringAttributes.ElementAt(i).GetValue(control) as string ?? "";
				i++;
			}

			return TextRenderer.MeasureText(text, control.Font).Width;
		}




	}
}