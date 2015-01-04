package math;

/** This will be the poster class to access the math library as static methods.
 */
public class DMath {
  public static boolean USE_CUDA = Boolean.parseBoolean(System.getProperty("use_cuda"));
  public DMath() {
    if(System.getProperty("use_cuda").equals("true"))
      USE_CUDA = true;
  }
  
  public static DMatrix createMatrix(int r, int c) {
    if(!USE_CUDA)
      return new BLASMatrix(r, c);
    else
      return new CUDAMatrix(r, c);
  }
  
  public static DMatrix createMatrix(int r, int c, double[] d) {
    if(!USE_CUDA)
      return new BLASMatrix(r, c, d);
    else
      return new CUDAMatrix(r, c, d);
  }

  public static DMatrix createZerosMatrix(int r, int c) {
    if(!USE_CUDA)
      return BLASMatrix.zeros(r, c);
    else
      return CUDAMatrix.zeros(r, c);
  }

  public static DMatrix createOnesMatrix(int r, int c) {
    if(!USE_CUDA)
      return BLASMatrix.ones(r, c);
    else
      return CUDAMatrix.ones(r, c);
  }

  public static DMatrix createRandnMatrix(int r, int c) {
    if(!USE_CUDA)
      return BLASMatrix.randn(r, c);
    else
      return CUDAMatrix.randn(r, c);
  }
}
