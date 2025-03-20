using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;

namespace GenericCuda
{
	public class CudaFftHandling
	{
		// ----- ATTRIBUTES ----- \\
		public CudaHandling CudaH;

		private PrimaryContext Ctx;



		// ----- OBJECTS ----- \\
		ListBox LogBox;




		// ----- LAMBDA ----- \\
		public CudaMemoryHandling? MemH => CudaH.MemH;
		public List<CUdeviceptr[]> Buffers => MemH?.Buffers.Keys.ToList() ?? [];
		public List<int[]> BufferSizes => MemH?.Buffers.Values.ToList() ?? [];

		public long[] IndexPointers => MemH?.IndexPointers ?? [];


		public int LogInterval => CudaH.LogInterval;


		// ----- CONSTRUCTOR ----- \\
		public CudaFftHandling(CudaHandling cudaH, PrimaryContext ctx, ListBox? logBox = null)
		{
			// Set attributes
			this.CudaH = cudaH;
			this.Ctx = ctx;
			this.LogBox = logBox ?? new ListBox();


			// Register events


		}




		// ----- METHODS ----- \\
		// Log
		public void Log(string message, string inner = "", int level = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("hh:mm:ss.fff") + "] ";
			msg += "<FFT> ";
			for (int i = 0; i < level; i++)
			{
				msg += " - ";
			}
			msg += message;
			if (!string.IsNullOrEmpty(inner))
			{
				msg += " (" + inner + ")";
			}
			if (update)
			{
				this.LogBox.Items[^1] = msg;
			}
			else
			{
				this.LogBox.Items.Add(msg);
			}
		}


		// Dispose
		public void Dispose()
		{
			// Dispose ...
		}


		// FFT
		public long PerformFFT(long indexPointer, bool keep = false)
		{
			// Abort if no memory handling
			if (MemH == null)
			{
				Log("No memory handling detected", "", 1);
				return 0;
			}

			// Get buffers & sizes
			CUdeviceptr[] buffers = MemH?[indexPointer] ?? [];
			int[] sizes = MemH?.GetSizesFromIndex(indexPointer) ?? [];

			// Abort if any int size is < 1 or if buffers are null
			if (buffers.LongLength == 0 || sizes.LongLength == 0 || sizes.Any(x => x < 1))
			{
				Log("Invalid buffers / sizes detected", "Count: " + buffers?.Length, 1);
				return indexPointer;
			}

			// Make buffers list for results (float2)
			CUdeviceptr[] results = new CUdeviceptr[buffers.LongLength];

			// Pre log
			this.Log("");

			// Perform FFT (forwards) for each buffer with the corresponding size
			for (int i = 0; i < buffers.LongLength; i++)
			{
				// Allocate result buffer
				results[i] = new CudaDeviceVariable<float2>(sizes[i]).DevicePointer;

				// Create plan
				CudaFFTPlan1D plan = new(sizes[i], cufftType.R2C, 1);
				
				// Execute plan
				plan.Exec(buffers[i], results[i]);
				
				// Dispose plan
				plan.Dispose();

				// Log progress every log interval
				if (i % LogInterval == 0)
				{
					Log("FFT performed", "Buffer: " + i + " / " + buffers.LongLength, 2, true);
				}
			}

			// Optionally keep buffers
			if (!keep)
			{
				MemH?.FreePointerGroup(indexPointer, true);
			}

			// Get index pointer for results
			long indexPointerResults = results.FirstOrDefault().Pointer;

			// Add results to MemH
			MemH?.Buffers.Add(results, sizes);

			// Log success
			Log("FFT completed", "Buffers: " + buffers.LongLength + ", Ptr: " + indexPointer, 1);


			// Return index pointer
			return indexPointerResults;
		}

		public long PerformIFFT(long indexPointer, bool keep = false)
		{
			// Abort if no memory handling
			if (MemH == null)
			{
				Log("No memory handling detected", "", 1);
				return 0;
			}

			// Get buffers & sizes
			CUdeviceptr[] buffers = MemH?[indexPointer] ?? [];
			int[] sizes = MemH?.GetSizesFromIndex(indexPointer) ?? [];

			// Abort if any int size is < 1 or if buffers are null
			if (buffers.LongLength == 0 || sizes.LongLength == 0 || sizes.Any(x => x < 1))
			{
				Log("Invalid buffers / sizes detected", "Count: " + buffers?.Length, 1);
				return indexPointer;
			}

			// Make buffers list for results (float2)
			CUdeviceptr[] results = new CUdeviceptr[buffers.LongLength];

			// Pre log
			this.Log("");

			// Perform FFT (forwards) for each buffer with the corresponding size
			for (int i = 0; i < buffers.LongLength; i++)
			{
				// Allocate result buffer
				results[i] = new CudaDeviceVariable<float>(sizes[i]).DevicePointer;

				// Create plan
				CudaFFTPlan1D plan = new(sizes[i], cufftType.C2R, 1);

				// Execute plan
				plan.Exec(buffers[i], results[i]);

				// Dispose plan
				plan.Dispose();

				// Log progress every log interval
				if (i % LogInterval == 0)
				{
					Log("I-FFT performed", "Buffer: " + i + " / " + buffers.LongLength, 2, true);
				}
			}

			// Optionally keep buffers
			if (!keep)
			{
				MemH?.FreePointerGroup(indexPointer, true);
			}

			// Get index pointer for results
			long indexPointerResults = results.FirstOrDefault().Pointer;

			// Add results to MemH
			MemH?.Buffers.Add(results, sizes);

			// Log success
			Log("FFT completed", "Buffers: " + buffers.LongLength + ", Ptr: " + indexPointer, 1);


			// Return index pointer
			return indexPointerResults;
		}
	}
}
