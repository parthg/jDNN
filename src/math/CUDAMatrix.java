package math;

import math.jcublas.SimpleCuBlas;
import random.RandomUtils;
import java.lang. AutoCloseable;
import jcuda.jcublas.JCublas;
import jcuda.Pointer;

import math.jblas.SimpleBlas;

// TODO: proper assertions
public class CUDAMatrix extends DMatrix implements AutoCloseable {
  public CUDAMatrix(int r, int c) {
    super(r, c);
    this.persist = false;
  }

  public CUDAMatrix(int r, int c, double[] d) {
    super(r, c, d);
    this.persist = false;
  }

  public CUDAMatrix(int r, int c, boolean _persist) {
    super(r, c);
    this.persist = _persist;
    if(this.persist) {
      this.cPointer = SimpleCuBlas.alloc(new double[r*c]);
    }
  }

  public CUDAMatrix(int r, int c, double[] d, boolean _persist) {
    super(r, c, d);
    this.persist = _persist;
    if(this.persist) {
      this.cPointer = SimpleCuBlas.alloc(d);
    }
  }

  public void close() {
//    System.err.printf("close() in CUDAMatrix\n");
    if(this.cPointer != null) {
//      System.out.printf("Releasing the CUDA Pointer\n");
      this.persist = false;
      JCublas.cublasFree(this.cPointer);
      this.cPointer = null;
    }
  }

  protected void finalize() {
//    System.err.printf("finalize() in CUDAMatrix()\n");
    if(this.cPointer != null) {
      this.close();
    }
  }

  public void copyHtoD() {
    if(System.getProperty("use_cuda").equals("true")) {
      if(this.persist == false)
        this.persist = true;
      if(this.cPointer != null) {
        JCublas.cublasFree(this.cPointer);
        this.cPointer = null;
      }
   
      this.cPointer = SimpleCuBlas.alloc(this.data());
    }
  }

  public void copyDtoH() {
    if(System.getProperty("use_cuda").equals("true")) {
      if (this.cPointer!=null) {
        SimpleCuBlas.getData(this,this.cPointer,Pointer.to(this.data()));
      }
    }
  }

  public void updateDeviceData() {
    if(this.cPointer!=null)
      SimpleCuBlas.updateData(this.cPointer, this.data);
  }
  
  public void updateDeviceData(double[] newData) {
    if(this.cPointer!=null)
      SimpleCuBlas.updateData(this.cPointer, newData);
  }
  
  public static DMatrix zeros(int r, int c) {
    return new CUDAMatrix(r, c);
  }

  public static DMatrix zeros(int r, int c, boolean _persist) {
    return new CUDAMatrix(r, c, _persist);
  }

  public static DMatrix ones(int r, int c) {
    DMatrix m = new CUDAMatrix(r, c);
    for(int i=0; i<r*c; i++)
      m.put(i, 1.0);
    return m;
  }

  public static DMatrix ones(int r, int c, boolean _persist) {
    DMatrix m = new CUDAMatrix(r, c);
    m.persist = _persist;
    for(int i=0; i<r*c; i++)
      m.put(i, 1.0);
    if(m.persist) {
      m.cPointer = SimpleCuBlas.alloc(m.data());
    }
    return m;
  }
  
  public static DMatrix randn(int r, int c) {
    DMatrix m = new CUDAMatrix(r, c);
    for (int i = 0; i < r * c; i++)
      m.put(i, RandomUtils.nextGaussian());
    return m;
  }
  
  public static DMatrix randn(int r, int c, boolean _persist) {
    DMatrix m = new CUDAMatrix(r, c);
    m.persist = _persist;
    for (int i = 0; i < r * c; i++)
      m.put(i, RandomUtils.nextGaussian());
    if(m.persist) {
      m.cPointer = SimpleCuBlas.alloc(m.data());
    }
    return m;
  }
  
  public DMatrix transpose() {
    return new CUDAMatrix(this.columns, this.rows, this.data);
  }
 

  // y = Ax+y
/*  public DMatrix addMuli(DMatrix A, DMatrix x) {
    SimpleCuBlas.gemv(A, x, this, 1.0, 1.0);
    return this;
  }*/



  // y = 1*x+y
  public DMatrix add(DMatrix other) {
    assert (this.length()==other.length());
    DMatrix m = new CUDAMatrix(this.rows, this.columns, this.data());
    SimpleCuBlas.axpy(1.0, other, m);
    return m;
  }
  public DMatrix addi(DMatrix other) {
    assert (this.length()==other.length());
//    System.out.printf("Using cuda blas\n");
    SimpleCuBlas.axpy(1.0, other, this);
    return this;
  }

  // y = a*X+b
  public DMatrix addi(double a, DMatrix other) {
    SimpleCuBlas.axpy(a, other, this);
    return this;
  }

  public DMatrix add(double v) {
    DMatrix m = DMath.createMatrix(this.rows(), this.columns(), this.toArray());
    for (int i = 0; i < this.length(); i++)
      m.put(i,(double) v+m.get(i));
    return m;
  }
  public DMatrix addi(double v) {
    for (int i = 0; i < this.length(); i++)
      this.put(i,(double) v+this.get(i));
    return this;
  }
  
  public DMatrix sub(DMatrix other) {
    assert this.length()==other.length() : System.out.printf("Length is not equal. %d - %d\n", this.length(), other.length());
    DMatrix m = new CUDAMatrix(this.rows(), this.columns(), this.toArray());
    SimpleCuBlas.axpy(-1.0, other, m);
    return m;
  }

  public DMatrix subi(DMatrix other) {
    assert (this.length()==other.length());
    SimpleCuBlas.axpy(-1.0, other, this);
    return this;
  }
  
  public DMatrix mul(DMatrix other) {
    assert (this.length()==other.length());
    DMatrix m = new CUDAMatrix(this.rows(), this.columns());
    SimpleCuBlas.mul(this, other, m);
    return m;
  }

  public DMatrix muli(DMatrix other) {
    assert (this.length()==other.length());
    SimpleCuBlas.mul(this, other, this);
    return this;
  }

  public DMatrix mul(double v) {
    DMatrix m = new CUDAMatrix(this.rows(), this.columns(), this.toArray());
    SimpleCuBlas.scal(m, v);
    return m;
  }
  public DMatrix muli(double v) {
    SimpleCuBlas.scal(this, v);
    return this;
  }

  public DMatrix pow(double v) {
    DMatrix m = new CUDAMatrix(this.rows(), this.columns(), this.toArray());
    SimpleCuBlas.pow(m, v);
    return m;
  }
  
  public DMatrix powi(double v) {
    SimpleCuBlas.pow(this, v);
    return this;
  }
  public DMatrix mmul(boolean tA, boolean tB, DMatrix B) {
//    assert (this.columns()==B.rows());
    DMatrix C = new CUDAMatrix(this.rows(), B.columns());
    return mmul(tA, tB, B, C);
  }

  public DMatrix mmul(DMatrix B) {
//    assert (this.columns()==B.rows());
    DMatrix C = new CUDAMatrix(this.rows(), B.columns());
    return mmul(false, false, B, C);
  }
  

  //result = this*other
  public DMatrix mmul(boolean tA, boolean tB, DMatrix B, DMatrix C) {
    //TODO: correct assertions in effect of tA and tB
    
    int m = tA?this.columns():this.rows();
    int n = tB?B.rows():B.columns();
    int k = tA?this.rows():this.columns();
    int kB = tB?B.columns():B.rows();
    assert (k==kB);
    if (C.rows != m || C.columns != n) {
      if (C != this && C != B) {
        C.resize(m, n);
      } else {
        System.err.printf("[ALERT] Should not resize result matrix because it is used in-place. But doing it anyway.\n");
      }
    }

    if (C == this || C == B) {
      /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
       * * allocating a temporary object on the side and copy the result later.
       * */
      DMatrix temp = new CUDAMatrix(m, n);
      if (m == 1) {
        SimpleCuBlas.gemv(tB, B, this, temp, 1.0, 0.0);
      } else {
        SimpleCuBlas.gemm(tA, tB, this, B, temp, 1.0, 0.0);
      }
      if(temp.rows()== C.rows() && temp.columns() == C.columns())
        SimpleCuBlas.copy(temp, C);
      else {
        C.resize(m , n);
        SimpleCuBlas.copy(temp, C);
      }
    } 
    else {
      if (m == 1) {
//        System.out.printf("calling gemv\n");
        SimpleCuBlas.gemv(tB, B, this, C, 1.0, 0.0);
      } else {

//        System.out.printf("calling gemm- HERE\n");
        SimpleCuBlas.gemm(tA, tB, this, B, C, 1.0, 0.0);
      }
    }
    return C;
  }
  
  public DMatrix mmul(DMatrix B, DMatrix C) {
//    assert (this.columns()==B.rows());
    return mmul(false, false, B, C);
    
  }
  
  public DMatrix mmuli(boolean tA, boolean tB, DMatrix B) {
//    assert (tA?this.rows:this.columns() == tB?tB.columns():B.rows());
    return mmul(tA, tB, B, this);
  }
  
  public DMatrix mmuli(DMatrix B) {
    assert (this.columns() == B.rows());
    return mmul(false, false, B, this);
  }


  public DMatrix fillWithArray(DMatrix other) {
    assert (this.length()%other.length()==0);
//    DMatrix m = DMath.createMatrix(this.rows(), this.columns());
    SimpleCuBlas.fillWithArray(other, this);
//    SimpleBlas.fillWithArray(other, this);
    return this;
  }
  
  public DMatrix sumRows() {
    DMatrix sum = DMath.createZerosMatrix(1, this.columns());
    DMatrix multiplier = DMath.createOnesMatrix(1, this.rows);
    SimpleCuBlas.gemv(false, this, multiplier, sum, 1.0, 0.0);
    return sum;
  }

  public DMatrix sumRows(int startRow, int howMany) {
    DMatrix sum = DMath.createMatrix(1, this.columns());
    DMatrix multiplier = DMath.createOnesMatrix(1, howMany);
    SimpleCuBlas.cust_gemv(false, this, multiplier, sum, 1.0, 0.0, startRow, howMany);
    return sum;
  }

  public DMatrix sumColumns() {
    DMatrix sum = DMath.createMatrix(this.rows, 1);
    DMatrix multiplier = DMath.createOnesMatrix(this.columns, 1);
    SimpleCuBlas.gemm(false, false, this, multiplier, sum, 1.0, 0.0);
    return sum;
  }
}
