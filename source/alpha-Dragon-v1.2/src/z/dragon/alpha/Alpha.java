/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.alpha;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.function.Function;
import static z.dragon.alpha.Alpha.NeuralParam.*;
import z.dragon.common.DragonCV;
import z.dragon.common.DragonFile;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateReader;
import z.dragon.common.state.State.StateWriter;
import z.dragon.data.Buffer;
import z.dragon.data.DataSet;
import z.dragon.data.TensorIter.TensorPair;
import z.dragon.data.Transform;
import z.dragon.data.container.AutoLoadContainer;
import z.dragon.data.container.AutoLoadContainer.Loader;
import z.dragon.data.container.ListContainer;
import z.dragon.engine.Engine;
import z.dragon.engine.EngineCore;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.cuda.impl.PinnedMempool;
import z.dragon.nn.loss.dim2.CrossEntropy;
import z.dragon.nn.loss.dim1.L2;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.loss.WeightedSummary;
import z.dragon.nn.loss.dim1.L1;
import z.dragon.nn.loss.dim1.BinaryCrossEntropy;
import z.dragon.nn.loss.dim1.SigmoidBinaryCrossEntropy;
import z.dragon.nn.loss.dim1.SmoothL1;
import z.dragon.nn.loss.dim2.SoftmaxCrossEntropy;
import z.dragon.nn.optim.Adam;
import z.dragon.nn.optim.AdamW;
import z.dragon.nn.optim.Adamax;
import z.dragon.nn.optim.Adamod;
import z.dragon.nn.optim.Momentum;
import z.dragon.nn.optim.RMSprop;
import z.dragon.nn.optim.SGD;
import z.dragon.nn.optim.lr_schedular.CosAnnealingLr;
import z.dragon.nn.optim.lr_schedular.ExponentialLr;
import z.dragon.nn.optim.lr_schedular.LambdaLr;
import z.dragon.nn.unit.complex.Sequence;
import z.dragon.nn.unit.furcation.tensor.Chunk;
import z.dragon.nn.unit.furcation.tensor.Split;
import z.dragon.nn.unit.reducer.tensor.Concat;
import z.dragon.nn.unit.reducer.math.LinearMean;
import z.dragon.nn.unit.reducer.math.LinearSummary;
import z.dragon.nn.unit.dual.blas.BatchMatMul;
import z.dragon.nn.unit.dual.blas.BatchMatMulT1;
import z.dragon.nn.unit.dual.blas.BatchMatMulT2;
import z.dragon.nn.unit.dual.math.Div;
import z.dragon.nn.unit.dual.blas.MatMul;
import z.dragon.nn.unit.dual.blas.MatMulT1;
import z.dragon.nn.unit.dual.blas.MatMulT2;
import z.dragon.nn.unit.dual.math.Quadratic2;
import z.dragon.nn.unit.simple.pool.adaptive.AdaptiveAvgPool2D;
import z.dragon.nn.unit.simple.pool.adaptive.AdaptiveMaxPool2D;
import z.dragon.nn.unit.simple.pool.AvgPool2D;
import z.dragon.nn.unit.simple.pool.AvgUnpool2D;
import z.dragon.nn.unit.simple.blas.Conv3D;
import z.dragon.nn.unit.simple.blas.Deconv3D;
import z.dragon.nn.unit.simple.blas.FullConnect;
import z.dragon.nn.unit.simple.pool.MaxPool2D;
import z.dragon.nn.unit.simple.math1.Abs;
import z.dragon.nn.unit.simple.affine.Affine;
import z.dragon.nn.unit.simple.math2.Arcsin;
import z.dragon.nn.unit.simple.math2.Arctan;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalSqBatchNorm;
import z.dragon.nn.unit.simple.math1.Cos;
import z.dragon.nn.unit.simple.math2.Cot;
import z.dragon.nn.unit.simple.math2.dropout.Dropout;
import z.dragon.nn.unit.simple.math2.Elu;
import z.dragon.nn.unit.simple.math2.Exp;
import z.dragon.nn.unit.simple.tensor.Flatten;
import z.dragon.nn.unit.simple.math2.HalfSin;
import z.dragon.nn.unit.simple.layernorm.LayerNorm;
import z.dragon.nn.unit.simple.math2.LeakyRelu;
import z.dragon.nn.unit.simple.math2.Linear;
import z.dragon.nn.unit.simple.math2.Log;
import z.dragon.nn.unit.simple.math1.Quadratic;
import z.dragon.nn.unit.simple.math2.Relu;
import z.dragon.nn.unit.simple.tensor.Rot180;
import z.dragon.nn.unit.simple.math2.Rpl;
import z.dragon.nn.unit.simple.math2.Sigmoid;
import z.dragon.nn.unit.simple.math1.Sin;
import z.dragon.nn.unit.simple.math2.Softplus;
import z.dragon.nn.unit.simple.math2.Sqrt;
import z.dragon.nn.unit.simple.math2.Tan;
import z.dragon.nn.unit.simple.math2.Tanh;
import z.dragon.nn.unit.simple.tensor.Transpose;
import z.dragon.nn.unit.simple.tensor.Reshape;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.dual.math.Linear2;
import z.dragon.nn.unit.reducer.math.QuadraticMean;
import z.dragon.nn.unit.reducer.math.QuadraticSummary;
import z.dragon.nn.unit.simple.batchnorm.SqBatchNorm;
import z.dragon.nn.unit.simple.math2.LogSoftmax;
import z.dragon.nn.unit.simple.math2.Softmax;
import z.dragon.nn.unit.simple.tensor.View;
import z.dragon.data.container.AutoLoadContainer.Triger;
import z.dragon.data.container.DataContainer;
import z.dragon.engine.EngineCore_ori;
import z.dragon.engine.memp.Memp1;
import z.dragon.engine.memp.Memp2;
import z.dragon.engine.memp.Memp3;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.simple.math2.bernouli.BernouliMul;
import z.util.math.vector.Vector;
import z.dragon.common.state.State.Stateful;
import z.dragon.common.state.State.StatefulTransformer;
import z.dragon.common.state.ZipState.ZipStateReader;
import z.dragon.common.state.ZipState.ZipStateWriter;
import z.dragon.dataset.Cifar10;
import z.dragon.dataset.Cifar100;
import z.dragon.data.FileFolder;
import z.dragon.data.ImageFolder;
import z.dragon.dataset.Minist;
import z.dragon.engine.Parameter;
import z.dragon.engine.cuda.impl.CudaDevice;
import z.dragon.nn.core.UnitCore;
import z.dragon.nn.core.dual.DualCore;
import z.dragon.nn.optim.SGDMN;
import z.dragon.nn.core.dual.blas.CoreBatchMatMul;
import z.dragon.nn.core.dual.blas.CoreBatchMatMulT1;
import z.dragon.nn.core.dual.blas.CoreBatchMatMulT2;
import z.dragon.nn.core.dual.blas.CoreMatMul;
import z.dragon.nn.core.dual.blas.CoreMatMulT1;
import z.dragon.nn.core.dual.blas.CoreMatMulT2;
import z.dragon.nn.core.dual.math.CoreDiv;
import z.dragon.nn.core.dual.math.CoreLinear2;
import z.dragon.nn.core.dual.math.CoreLinear2Row;
import z.dragon.nn.core.dual.math.CoreLinear2_Elu;
import z.dragon.nn.core.dual.math.CoreLinear2_Gelu;
import z.dragon.nn.core.dual.math.CoreLinear2_LeakyRelu;
import z.dragon.nn.core.dual.math.CoreLinear2_Relu;
import z.dragon.nn.core.dual.math.CoreLinear2_Sigmoid;
import z.dragon.nn.core.dual.math.CoreLinear2_Softplus;
import z.dragon.nn.core.dual.math.CoreLinear2_Tanh;
import z.dragon.nn.core.dual.math.CoreQuadratic2;
import z.dragon.nn.core.dual.math.CoreQuadratic2Center;
import z.dragon.nn.core.dual.math.CoreQuadratic2Row;
import z.dragon.nn.core.furcation.FurcationCore;
import z.dragon.nn.core.furcation.tensor.CoreChunk;
import z.dragon.nn.core.furcation.tensor.CoreSplit;
import z.dragon.nn.core.reducer.ReducerCore;
import z.dragon.nn.core.reducer.math.CoreLinearMean;
import z.dragon.nn.core.reducer.math.CoreLinearSummary;
import z.dragon.nn.core.reducer.math.CoreQuadraticMean;
import z.dragon.nn.core.reducer.math.CoreQuadraticSummary;
import z.dragon.nn.core.reducer.tensor.CoreConcat;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleFunction;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm;
import z.dragon.nn.unit.simple.math1.Csc;
import z.dragon.nn.unit.simple.math1.Sec;
import z.dragon.nn.core.simple.math1.CoreAbs;
import z.dragon.nn.core.simple.math1.CoreCos;
import z.dragon.nn.core.simple.math1.CoreCsc;
import z.dragon.nn.core.simple.math1.CoreGelu;
import z.dragon.nn.core.simple.math1.CoreQuadratic;
import z.dragon.nn.core.simple.math1.CoreSec;
import z.dragon.nn.core.simple.math1.CoreSin;
import z.dragon.nn.core.simple.math2.CoreArcsin;
import z.dragon.nn.core.simple.math2.CoreArctan;
import z.dragon.nn.core.simple.math2.CoreClip;
import z.dragon.nn.core.simple.math2.CoreCot;
import z.dragon.nn.core.simple.math2.CoreElu;
import z.dragon.nn.core.simple.math2.CoreExp;
import z.dragon.nn.core.simple.math2.CoreHalfSin;
import z.dragon.nn.core.simple.math2.CoreHardSigmoid;
import z.dragon.nn.core.simple.math2.CoreLeakyRelu;
import z.dragon.nn.core.simple.math2.CoreLinear;
import z.dragon.nn.core.simple.math2.CoreLogSoftmax;
import z.dragon.nn.core.simple.math2.CoreLog;
import z.dragon.nn.core.simple.math2.CoreMax;
import z.dragon.nn.core.simple.math2.CoreMin;
import z.dragon.nn.core.simple.math2.CoreRelu;
import z.dragon.nn.core.simple.math2.CoreReluN;
import z.dragon.nn.core.simple.math2.CoreRpl;
import z.dragon.nn.core.simple.math2.CoreSigmoid;
import z.dragon.nn.core.simple.math2.CoreSoftmax;
import z.dragon.nn.core.simple.math2.CoreSoftplus;
import z.dragon.nn.core.simple.math2.CoreSqrt;
import z.dragon.nn.core.simple.math2.CoreTan;
import z.dragon.nn.core.simple.math2.CoreTanh;
import z.dragon.nn.core.simple.pool.CoreAvgPool1D;
import z.dragon.nn.core.simple.pool.adaptive.CoreAdaptiveAvgPool2D;
import z.dragon.nn.core.simple.pool.adaptive.CoreAdaptiveMaxPool2D;
import z.dragon.nn.core.simple.pool.CoreAvgPool2D;
import z.dragon.nn.core.simple.pool.CoreAvgUnpool2D;
import z.dragon.nn.core.simple.pool.CoreMaxPool1D;
import z.dragon.nn.core.simple.pool.CoreMaxPool2D;
import z.dragon.nn.core.simple.pool.adaptive.CoreAdaptiveAvgPool1D;
import z.dragon.nn.core.simple.pool.adaptive.CoreAdaptiveMaxPool1D;
import z.dragon.nn.core.simple.tensor.CoreCrop;
import z.dragon.nn.core.simple.tensor.CoreExpand;
import z.dragon.nn.unit.simple.tensor.Pad;
import z.dragon.nn.unit.simple.tensor.Trim;
import z.dragon.nn.core.simple.tensor.CoreFlatten;
import z.dragon.nn.core.simple.tensor.CoreView;
import z.dragon.nn.core.simple.tensor.CorePad;
import z.dragon.nn.core.simple.tensor.CoreReshape;
import z.dragon.nn.core.simple.tensor.CoreRot180;
import z.dragon.nn.core.simple.tensor.CoreTranspose;
import z.dragon.nn.core.simple.tensor.CoreTrim;
import z.dragon.nn.optim.RAdam;
import z.dragon.nn.unit.dual.DualFunction;
import z.dragon.nn.unit.dual.math.Linear2Row;
import z.dragon.nn.unit.dual.math.Linear2_Elu;
import z.dragon.nn.unit.dual.math.Linear2_Gelu;
import z.dragon.nn.unit.dual.math.Linear2_LeakyRelu;
import z.dragon.nn.unit.dual.math.Linear2_Relu;
import z.dragon.nn.unit.dual.math.Linear2_Sigmoid;
import z.dragon.nn.unit.dual.math.Linear2_Softplus;
import z.dragon.nn.unit.dual.math.Linear2_Tanh;
import z.dragon.nn.unit.dual.math.Quadratic2Center;
import z.dragon.nn.unit.dual.math.Quadratic2Row;
import z.dragon.nn.unit.furcation.FurcateFunction;
import z.dragon.nn.unit.reducer.ReduceFunction;
import z.dragon.nn.unit.simple.affine.Affine_Elu;
import z.dragon.nn.unit.simple.affine.Affine_Gelu;
import z.dragon.nn.unit.simple.affine.Affine_LeakyRelu;
import z.dragon.nn.unit.simple.affine.Affine_Relu;
import z.dragon.nn.unit.simple.affine.Affine_Sigmoid;
import z.dragon.nn.unit.simple.affine.Affine_Softplus;
import z.dragon.nn.unit.simple.affine.Affine_Tanh;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Elu;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Gelu;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_LeakyRelu;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Relu;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Sigmoid;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Softplus;
import z.dragon.nn.unit.simple.batchnorm.BatchNorm_Tanh;
import z.dragon.nn.unit.simple.blas.Conv2D;
import z.dragon.nn.unit.simple.blas.Deconv2D;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Elu;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Gelu;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_LeakyRelu;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Relu;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Sigmoid;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Softplus;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm_Tanh;
import z.dragon.nn.unit.simple.math1.Gelu;
import z.dragon.nn.unit.simple.math2.Clip;
import z.dragon.nn.unit.simple.math2.HardSigmoid;
import z.dragon.nn.unit.simple.math2.bernouli.LeakyRelu_BernouliMul;
import z.dragon.nn.unit.simple.math2.dropout.LeakyRelu_Dropout;
import z.dragon.nn.unit.simple.math2.Max;
import z.dragon.nn.unit.simple.math2.Min;
import z.dragon.nn.unit.simple.math2.ReluN;
import z.dragon.nn.unit.simple.math2.bernouli.Elu_BernouliMul;
import z.dragon.nn.unit.simple.math2.bernouli.Gelu_BernouliMul;
import z.dragon.nn.unit.simple.math2.dropout.Elu_Dropout;
import z.dragon.nn.unit.simple.math2.dropout.Gelu_Dropout;
import z.dragon.nn.unit.simple.math2.bernouli.Relu_BernouliMul;
import z.dragon.nn.unit.simple.math2.bernouli.Sigmoid_BernouliMul;
import z.dragon.nn.unit.simple.math2.bernouli.Softplus_BernouliMul;
import z.dragon.nn.unit.simple.math2.bernouli.Tanh_BernouliMul;
import z.dragon.nn.unit.simple.math2.dropout.Relu_Dropout;
import z.dragon.nn.unit.simple.math2.dropout.Sigmoid_Dropout;
import z.dragon.nn.unit.simple.math2.dropout.Softplus_Dropout;
import z.dragon.nn.unit.simple.pool.AvgPool1D;
import z.dragon.nn.unit.simple.pool.MaxPool1D;
import z.dragon.nn.unit.simple.pool.adaptive.AdaptiveAvgPool1D;
import z.dragon.nn.unit.simple.pool.adaptive.AdaptiveMaxPool1D;
import z.dragon.nn.unit.simple.tensor.Crop;
import z.dragon.nn.unit.simple.tensor.Expand;

/**
 *
 * @author Gilgamesh
 */
public final class Alpha {
    public final long MEM_1GB = Engines.MEM_1GB;
    public final long MEM_1MB = Engines.MEM_1MB;
    
    //<editor-fold defaultstate="collapsed" desc="member-parameters"> 
    public final Engines engine = Engines.engine;
    public final UnitBuilder nn = UnitBuilder.nn;
    public final Loss loss = Loss.loss;
    public final Optim optim = Optim.optim;
    public final Datas data = Datas.data;
    public final Stats stat = Stats.stat;
    public final UnitFunctional F = UnitFunctional.F;
    
    public final DragonCV cv = DragonCV.instance();
    public final DragonFile fl = DragonFile.instance();
    //</editor-fold>
    
    private Alpha() {}
    
    public static final Alpha alpha = new Alpha();
    public static final String version = "alpha-v1.2";
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("Dragon-Alpha-v1.2 { ");
        sb.append("\n\tAuthor: 张智溢 Gilgamesh.CN");
        sb.append("\n\tDate: 2024/8/30");
        sb.append("\n\tAnywhere, try your best! }");
        return sb.toString();
    }
    
    //<editor-fold defaultstate="collapsed" desc="alpha: string && print(args)">
    //<editor-fold defaultstate="collapsed" desc="toString: float">
    public String format(float f) { 
        return (f > 1e-4f) || (-f > 1e-4f)? 
                String.format("%6f", f):
                String.format("%6e", f); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, float[])">
    public void append(StringBuilder sb, float[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    
    public void append(StringBuilder sb, float[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, float[][][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]); 
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, prefix + " ", X[i]); } 
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string(float[])">
    public String str(float[] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(float[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toString: int">
    public String format(int d) { return Integer.toString(d); }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, int[])">
    public void append(StringBuilder sb, int[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    
    public void append(StringBuilder sb, int[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, int[][][][] X) {
        if(X == null) return;
        String start = "\n " + prefix, next_pre = prefix + " ";
        sb.append('['); append(sb, next_pre, X[0]);
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, next_pre, X[i]); } 
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="string(int[])">
    public String str(int[] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(int[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 3;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toString: byte">
    public String format(byte d) { return String.format("%3d", d); }
    
    //<editor-fold defaultstate="collapsed" desc="append(sb, byte[])">
    public void append(StringBuilder sb, byte[] X) {
        if(X == null) return;
        sb.append('[').append(format(X[0]));
        for(int i=1; i<X.length; i++) sb.append(", ").append(format(X[i]));
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][][] X)  { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    
    public void append(StringBuilder sb, byte[][][][] X) { append(sb, "", X); }
    public void append(StringBuilder sb, String prefix, byte[][][][] X) {
        if(X == null) return;
        sb.append('['); append(sb, X[0]); 
        String start = "\n " + prefix;
        for(int i=1; i<X.length; i++) { sb.append(start); append(sb, X[i]); }
        sb.append(']');
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sting(byte[])">
    public String str(byte[] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
    
    public String str(byte[][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        StringBuilder sb = new StringBuilder(size);
        append(sb, X);
        return sb.toString();
    }
      public String str(byte[][][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    
    public String str(byte[][][][] X) {
        if(X == null) return "null";
        int size = X.length << 2;
        if(X[0] != null) size *= X[0].length;
        if(X[0][0] != null) size *= X[0][0].length;
        if(X[0][0][0] != null) size *= X[0][0][0].length;
        StringBuilder sb = new StringBuilder(size);
        Alpha.this.append(sb, X);
        return sb.toString();
    }
    //</editor-fold>
    //</editor-fold>
    
    public String string(Object arg) {
        //array 1D--------------------------------------------------------------
        if(arg instanceof   float[]) return str((float[]) arg);
        if(arg instanceof    byte[]) return str((byte[])  arg);
        if(arg instanceof     int[]) return str((int[])   arg);
        
        //array 2D--------------------------------------------------------------
        if(arg instanceof   float[][]) return str((float[][]) arg);
        if(arg instanceof    byte[][]) return str((byte[][])  arg);
        if(arg instanceof     int[][]) return str((int[][])   arg);
        
        //array 3D--------------------------------------------------------------
        if(arg instanceof   float[][][]) return str((float[]) arg);
        if(arg instanceof    byte[][][]) return str((byte[])  arg);
        if(arg instanceof     int[][][]) return str((int[])   arg);
        
        //array 3D--------------------------------------------------------------
        if(arg instanceof   float[][][][]) return str((float[]) arg);
        if(arg instanceof    byte[][][][]) return str((byte[])  arg);
        if(arg instanceof     int[][][][]) return str((int[])   arg);
        
        return Objects.toString(arg);
    }
    
    private boolean __print_array1D(Object arg) {
        if(arg instanceof   float[]) { out.print(str((float[]) arg)); return true; }
        if(arg instanceof    byte[]) { out.print(str((byte[])  arg)); return true; }
        if(arg instanceof     int[]) { out.print(str((int[])   arg)); return true; }
        return false;
    }
    
    private boolean __print_array2D(Object arg) {
        if(arg instanceof   float[][]) { out.print(str((float[][])arg)); return true; }
        if(arg instanceof    byte[][]) { out.print(str( (byte[][])arg)); return true; }
        if(arg instanceof     int[][]) { out.print(str(  (int[][])arg)); return true; }
        return false;
    }
    
    private boolean __print_array3D(Object arg) {
        if(arg instanceof float[][][]) { out.print(str((float[][][])arg)); return true;}
        if(arg instanceof  byte[][][]) { out.print(str(( byte[][][])arg)); return true;}
        if(arg instanceof   int[][][]) { out.print(str(  (int[][][])arg)); return true;}
        return false;
    }
    
    private boolean __print_array4D(Object arg) {
        if(arg instanceof float[][][][]) { out.print(str((float[][][][])arg)); return true;}
        if(arg instanceof  byte[][][][]) { out.print(str( (byte[][][][])arg)); return true;}
        if(arg instanceof   int[][][][]) { out.print(str(  (int[][][][])arg)); return true;}
        return false;
    }
    
    public static PrintStream out = System.out;
    public Alpha print(Object... args) {
        if(args == null) { out.println("null"); return this; }
        if(__print_array2D(args)) { out.println(); return this; }
        if(__print_array3D(args)) { out.println(); return this; }
        if(__print_array4D(args)) { out.println(); return this; }
        
        boolean flag = false;
        for(Object arg : args) {
            if(__print_array1D(arg)) continue;
            if(__print_array2D(arg)) continue;
            if(__print_array3D(arg)) continue;
            if(__print_array4D(arg)) continue;
            out.print(Objects.toString(arg));
            if(flag) out.print(", "); else flag = true;
        }
        
        out.println(); 
        return this;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="parallel_class: Line">
    static final ThreadFactory daemonThreadFactory = (Runnable r) -> { Thread t = new Thread(r); t.setDaemon(true); return t; };
    static final ExecutorService exec = Executors.newFixedThreadPool(4, daemonThreadFactory); 
    
    public static class Line<T> 
    {
        private final Future<T> ft;
        
        public Line(Future<T> ft) { this.ft = ft; }
        
        public T c() {
            try { return ft.get(); }
            catch(InterruptedException | ExecutionException e) {
                throw new RuntimeException(e); 
            }
        }
    }
    
    public <T> Line<T> line(Callable<T> call) {  return new Line<>(exec.submit(call));  }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ALPHA_HOME">
    static String alpha_home;
    public String home() { return alpha_home; }
    
    public synchronized void home(String home) {
        if(home == null || home.isEmpty()) throw new NullPointerException("ALPHA_HOME can't be empty");
        
        alpha_home = home;//alpha_home是文件夹位置的标识，后面不能加 \
        if(alpha_home.endsWith("\\")) alpha_home = alpha_home.substring(0, alpha_home.length() - 1);
        
        //load native lib: CudaFloat32EngineBase--------------------------------
        if (!CudaFloat32EngineBase.__NATIVE_LOAD__()) {
            try {
                CudaFloat32EngineBase.load_native_lib(alpha_home);
                CudaFloat32EngineBase.__SET_NATIVE_LOAD__(true);
                System.out.println("[" + version + "]: CudaFloat32-nativeLib has been loaded");
            }
            catch(Exception e) {
                System.err.println("[" + version + "]: Fail to load CudaFloat32-nativeLib");
                throw new RuntimeException(e);
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Engines">
    public static class Engines 
    {
        protected Engines() {}
        
        public static final long MEM_1GB = (1L) << 30;
        public static final long MEM_1MB = (1L) << 20;
        public static final Engines engine = new Engines();
        
        public static final int default_cuda_stream_pool_size = 64;
        public static final long default_max_memory_size = 2 * MEM_1GB;
        public static final long default_max_transfer_buf_size = 512 * MEM_1MB;
        
        //<editor-fold defaultstate="collapsed" desc="create: mempool">
        public Memp1 memp1(long maxMemorySize) { return new Memp1(maxMemorySize); }
        public Memp2 memp2(long maxMemorySize) { return new Memp2(maxMemorySize); }
        public Memp3 memp3(long maxMemorySize) { return new Memp3(maxMemorySize); }
        
        public Memp1 memp1() { return new Memp1(default_max_memory_size); }
        public Memp2 memp2() { return new Memp2(default_max_memory_size); }
        public Memp3 memp3() { return new Memp3(default_max_memory_size); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="cufa-float32-old">
        public Engine old_cuda_float32(int deviceId, 
                int streamPool_maxsize,
                long maxMemorySize,
                long maxTransferBufSize,//PinnedMemoryPool
                boolean check) 
        {
            CudaFloat32EngineBase base = new CudaFloat32EngineBase(deviceId, streamPool_maxsize);
            if(maxTransferBufSize > 0) base.buf_mempool(new PinnedMempool(maxTransferBufSize));
        
            EngineCore core = new EngineCore_ori(maxMemorySize, check).engineBase(base);
            return new Engine().engineCore(core);
        }

        public Engine old_cuda_float32(int deviceId, long maxMemorySize, long maxBufSize, boolean check) {
            return Engines.this.old_cuda_float32(deviceId, 
                    default_cuda_stream_pool_size,
                    maxMemorySize, 
                    maxBufSize, 
                    check);
        }
        
        public Engine old_cuda_float32(int deviceId, long maxMemorySize, long maxBufSize) {
            return Engines.this.old_cuda_float32(deviceId,
                    default_cuda_stream_pool_size, 
                    maxMemorySize,
                    maxBufSize,
                    true);
        }
        
         public Engine old_cuda_float32(int deviceId, long maxMemorySize) {
            return Engines.this.old_cuda_float32(deviceId,
                    default_cuda_stream_pool_size, 
                    maxMemorySize,
                    default_max_transfer_buf_size,
                    true);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="cufa-float32">
          public Engine cuda_float32(CudaDevice device, 
                int streamPool_maxsize,
                Mempool mempool, 
                long max_trans_buf_size,//PinnedMemoryPool
                boolean check) 
        {
            CudaFloat32EngineBase base = new CudaFloat32EngineBase(device, streamPool_maxsize);
            if(max_trans_buf_size > 0) base.buf_mempool(new PinnedMempool(max_trans_buf_size));
            EngineCore core = new EngineCore(mempool, check).engineBase(base);
            return new Engine().engineCore(core);
        }
         
        public Engine cuda_float32(int deviceId, 
                int streamPool_maxsize,
                Mempool mempool, 
                long max_trans_buf_size,//PinnedMemoryPool
                boolean check) 
        {
            CudaFloat32EngineBase base = new CudaFloat32EngineBase(deviceId, streamPool_maxsize);
            if(max_trans_buf_size > 0) base.buf_mempool(new PinnedMempool(max_trans_buf_size));
            EngineCore core = new EngineCore(mempool, check).engineBase(base);
            return new Engine().engineCore(core);
        }
        
        public Engine cuda_float32(int deviceId, Mempool memp, long max_trans_buf_size) {
            return cuda_float32(deviceId, 
                    default_cuda_stream_pool_size,
                    memp,
                    max_trans_buf_size, 
                    true);
        }
        
       public Engine cuda_float32(int deviceId, Mempool memp) {
            return cuda_float32(deviceId, 
                    default_cuda_stream_pool_size,
                    memp, 
                    default_max_transfer_buf_size,
                    true);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: NeuralParam">
    public static class NeuralParam {
        public static boolean sp_math2_inplace = true;
        public static boolean sp_tensor_inplace = true;
        public static boolean sp_affine_inplace = true;
        
        public static float leakyRelu_neg_slope = 0.01f;
        public static float elu_alpha = 1.0f;
        public static float elu_neg_slope = 0.01f;
        
        public static boolean batchNorm_affine = true;
        public static float batchNorm_beta1 = 0.9f;
        public static float batchNorm_beta2 = 0.9f;
        public static float batchNorm_eps = 1e-5f;
        
        public static boolean layerNorm_affine = true;
        public static float layerNorm_eps = 1e-5f;
        
        public static boolean avgpool2D_igpad = false;
        public static boolean avgpool1D_igpad = false;
        public static boolean avgunpool2D_ignore_padding = false;
         
        public static boolean dl_likeX1 = true;
    }
    //</editor-fold>       
    
    //<editor-fold defaultstate="collapsed" desc="class: UnitBuilder">
    public static class UnitBuilder {
        protected UnitBuilder() {}
        public static final UnitBuilder nn = new UnitBuilder();
        
        //<editor-fold defaultstate="collapsed" desc="create: fusion">
        public Affine_Relu affine_relu(Affine afi, Relu af) { return new Affine_Relu(afi.inplace() && af.inplace(), afi.param_dim()); }
        public Affine_LeakyRelu affine_leakyRelu(Affine afi, LeakyRelu af) { return new Affine_LeakyRelu(afi.inplace() && af.inplace(), af.negative_slope(), afi.param_dim()); }
        public Affine_Elu affine_elu(Affine afi, Elu af) { return new Affine_Elu(afi.inplace() && af.inplace(), af.alpha(), af.negative_slope(), afi.param_dim()); }
        public Affine_Softplus affine_softplus(Affine afi, Softplus af) { return new Affine_Softplus(afi.inplace() && af.inplace(), afi.param_dim()); }
        public Affine_Gelu affine_gelu(Affine afi, Gelu af) { return new Affine_Gelu(afi.inplace(), afi.param_dim()); }
        public Affine_Sigmoid affine_sigmoid(Affine afi, Sigmoid af) { return new Affine_Sigmoid(afi.inplace() && af.inplace(), afi.param_dim()); }
        public Affine_Tanh affine_tanh(Affine afi, Tanh af) { return new Affine_Tanh(afi.inplace() && af.inplace(), afi.param_dim()); }
        
        public GlobalBatchNorm_Relu global_batchNorm_relu(GlobalBatchNorm bn, Relu af) { 
            return new GlobalBatchNorm_Relu(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        public GlobalBatchNorm_LeakyRelu global_batchNorm_leakyRelu(GlobalBatchNorm bn, LeakyRelu af) { 
            return new GlobalBatchNorm_LeakyRelu(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), af.negative_slope(), bn.param_dim()); 
        }
        public GlobalBatchNorm_Elu global_batchNorm_elu(GlobalBatchNorm bn, Elu af) {
            return new GlobalBatchNorm_Elu(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), af.alpha(), af.negative_slope(), bn.param_dim());
        }
        public GlobalBatchNorm_Softplus global_batchNorm_softplus(GlobalBatchNorm bn, Softplus af) {
            return new GlobalBatchNorm_Softplus(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim());
        }
        public GlobalBatchNorm_Gelu global_batchNorm_gelu(GlobalBatchNorm bn, Gelu af) {
            return new GlobalBatchNorm_Gelu(bn.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim());
        }
        public GlobalBatchNorm_Sigmoid global_batchNorm_sigmoid(GlobalBatchNorm bn, Sigmoid af) {
            return new GlobalBatchNorm_Sigmoid(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim());
        }
        public GlobalBatchNorm_Tanh global_batchNorm_tanh(GlobalBatchNorm bn, Tanh af) {
            return new GlobalBatchNorm_Tanh(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim());
        }
        
        public BatchNorm_Relu batchNorm_relu(BatchNorm bn, Relu af) { 
            return new BatchNorm_Relu(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        public BatchNorm_LeakyRelu batchNorm_leakyRelu(BatchNorm bn, LeakyRelu af) {
            return new BatchNorm_LeakyRelu(bn.inplace() && af.inplace(), bn.affine(),
                    bn.beta1(), bn.beta2(), bn.eps(), af.negative_slope(), bn.param_dim());
        }
        public BatchNorm_Elu batchNorm_elu(BatchNorm bn, Elu af) {
            return new BatchNorm_Elu(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), af.alpha(), af.negative_slope(), bn.param_dim()); 
        }
        public BatchNorm_Softplus batchNorm_softplus(BatchNorm bn, Softplus af) { 
            return new BatchNorm_Softplus(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        public BatchNorm_Gelu batchNorm_gelu(BatchNorm bn, Gelu af) { 
            return new BatchNorm_Gelu(bn.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        public BatchNorm_Sigmoid batchNorm_sigmoid(BatchNorm bn, Sigmoid af) { 
            return new BatchNorm_Sigmoid(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        public BatchNorm_Tanh batchNorm_tanh(BatchNorm bn, Tanh af) {
            return new BatchNorm_Tanh(bn.inplace() && af.inplace(), bn.affine(), 
                    bn.beta1(), bn.beta2(), bn.eps(), bn.param_dim()); 
        }
        
        public Linear2_Relu linear2_relu(Linear2 l2, Relu af) { return new Linear2_Relu(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma()); }
        public Linear2_LeakyRelu linear2_leakyRelu(Linear2 l2, LeakyRelu af) { return new Linear2_LeakyRelu(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma(), af.negative_slope()); }
        public Linear2_Elu linear2_Elu(Linear2 l2, Elu af) { return new Linear2_Elu(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma(), af.alpha(), af.negative_slope()); }
        public Linear2_Softplus linear2_softplus(Linear2 l2, Softplus af) { return new Linear2_Softplus(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma()); }
        public Linear2_Gelu linear2_gelu(Linear2 l2, Gelu af) { return new Linear2_Gelu(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma()); }
        public Linear2_Sigmoid linear2_sigmoid(Linear2 l2, Sigmoid af) { return new Linear2_Sigmoid(l2.likeX1(), l2.alpha(), l2.beta(), l2.gamma()); }
        public Linear2_Tanh linear2_tanh(Linear2 linear2, Tanh af) { return new Linear2_Tanh(linear2.likeX1(), linear2.alpha(), linear2.beta(), linear2.gamma()); }
        
        public Relu_Dropout relu_dropout(Relu af, Dropout dp) { return new Relu_Dropout(af.inplace() && dp.inplace(), dp.nonzero_percent()); }
        public LeakyRelu_Dropout leakyRelu_dropout(LeakyRelu af, Dropout dp) { return new LeakyRelu_Dropout(af.inplace() && dp.inplace(), af.negative_slope(), dp.nonzero_percent()); }
        public Elu_Dropout elu_dropout(Elu af, Dropout dp) { return new Elu_Dropout(af.inplace() && dp.inplace(), af.alpha(), af.negative_slope(), dp.nonzero_percent()); }
        public Softplus_Dropout softplus_dropout(Softplus af, Dropout dp) { return new Softplus_Dropout(af.inplace() && dp.inplace(), dp.nonzero_percent()); }
        public Gelu_Dropout gelu_dropout(Gelu af, Dropout dp) { return new Gelu_Dropout(dp.inplace(), dp.nonzero_percent()); }
        public Sigmoid_Dropout softplus_dropout(Sigmoid af, Dropout dp) { return new Sigmoid_Dropout(af.inplace() && dp.inplace(), dp.nonzero_percent()); }
       
        public Relu_BernouliMul relu_bernouliMul(Relu af, BernouliMul bm) { return new Relu_BernouliMul(af.inplace() && bm.inplace(), bm.p(), bm.v1(), bm.v2()); }
        public LeakyRelu_BernouliMul leakyRelu_bernouliMul(LeakyRelu af, BernouliMul bm) { return new LeakyRelu_BernouliMul(af.inplace() && bm.inplace(), af.negative_slope(), bm.p(), bm.v1(), bm.v2()); }
        public Elu_BernouliMul elu_bernouliMul(Elu af, BernouliMul bm) { return new Elu_BernouliMul(af.inplace() && bm.inplace(), af.alpha(), af.negative_slope(), bm.p(), bm.v1(), bm.v2()); }
        public Softplus_BernouliMul softplus_bernouliMul(Softplus af, BernouliMul bm) { return new Softplus_BernouliMul(af.inplace() && bm.inplace(), bm.p(), bm.v1(), bm.v2()); }
        public Gelu_BernouliMul gelu_bernouliMul(Gelu af, BernouliMul bm) { return new Gelu_BernouliMul(bm.inplace(), bm.p(), bm.v1(), bm.v2()); }
        public Sigmoid_BernouliMul softplus_bernouliMul(Sigmoid af, BernouliMul bm) { return new Sigmoid_BernouliMul(af.inplace() && bm.inplace(), bm.p(), bm.v1(), bm.v2()); }
        public Tanh_BernouliMul tanh_bernouliMul(Tanh af, BernouliMul bm) { return new Tanh_BernouliMul(af.inplace() && bm.inplace(), bm.p(), bm.v1(), bm.v2()); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.math1">
        public Abs abs() { return new Abs(1.0f, 0.0f); }
        public Abs abs(float alpha, float beta) { return new Abs(alpha, beta); }
        
        public Sin sin() { return new Sin(1.0f, 0.0f); }
        public Sin sin(float alpha, float beta) { return new Sin(alpha, beta); }
        
        public Cos cos() { return new Cos(1.0f, 0.0f);}
        public Cos cos(float alpha, float beta) { return new Cos(alpha, beta); }
        
        public Csc csc() { return new Csc(1.0f, 0.0f); }
        public Csc csc(float alpha, float beta) { return new Csc(alpha, beta); }
        public Sec sec() { return new Sec(1.0f, 0.0f); }
        public Sec sec(float alpha, float beta) { return new Sec(alpha, beta); }
        
        public Quadratic square() { return new Quadratic(1.0f, 0.0f, 0.0f); }
        public Quadratic square(float alpha) { return new Quadratic(alpha, 0.0f, 0.0f); }
        public Quadratic quadratic(float alpha, float beta, float gamma){
            return new Quadratic(alpha, beta, gamma);
        }
        
        public Gelu gelu() { return new Gelu(); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.math2">
        public Max max(float vmax) { return new Max(sp_math2_inplace, 1.0f, 0.0f, vmax); }
        public Max max(float alpha, float beta, float vmax) { return new Max(sp_math2_inplace, alpha, beta, vmax); }
        public Max max(boolean inplace, float vmax) { return new Max(inplace, 1.0f, 0.0f, vmax); }
        public Max max(boolean inplace, float alpha, float beta, float vmax) { return new Max(inplace, alpha, beta, vmax); }
        
        public Min min(float vmin) { return new Min(sp_math2_inplace, 1.0f, 0.0f, vmin); }
        public Min min(float alpha, float beta, float vmin) { return new Min(sp_math2_inplace, alpha, beta, vmin); }
        public Min min(boolean inplace, float vmin) { return new Min(inplace, 1.0f, 0.0f, vmin); }
        public Min min(boolean inplace, float alpha, float beta, float vmin) { return new Min(inplace, alpha, beta, vmin); }
        
        public Clip clip(float vmin, float vmax) { return new Clip(sp_math2_inplace, 1.0f, 0.0f, vmin, vmax); }
        public Clip clip(float alpha, float beta, float vmin, float vmax) {
            return new Clip(sp_math2_inplace, alpha, beta, vmin, vmax);
        }
        public Clip clip(boolean inplace, float vmin, float vmax) { return new Clip(inplace, 1.0f, 0.0f, vmin, vmax); }
        public Clip clip(boolean inplace, float alpha, float beta, float vmin, float vmax) {
            return new Clip(inplace, alpha, beta, vmin, vmax);
        }
        
        public Rpl rpl() { return new Rpl(sp_math2_inplace, 1.0f, 0.0f, 0.0f); }
        public Rpl rpl(float alpha, float beta, float gamma) { return new Rpl(sp_math2_inplace, alpha, beta, gamma); }
        public Rpl rpl(boolean inplace) { return new Rpl(inplace, 1.0f, 0.0f, 0.0f); }
        public Rpl rpl(boolean inplace, float alpha, float beta, float gamma) { return new Rpl(inplace, alpha, beta, gamma); }
        
        public Linear sadd(float C) { return new Linear(sp_math2_inplace, 1.0f, C); }
        public Linear ssub(float C) { return new Linear(sp_math2_inplace, 1.0f, -C); }
        public Linear smul(float C) { return new Linear(sp_math2_inplace, C, 0.0f); }
        public Linear sdiv(float C) { return new Linear(sp_math2_inplace, (1.0f / C), 0.0f); }
        public Linear linear(float alpha, float beta) { return new Linear(sp_math2_inplace, alpha, beta); }
        
        public Linear sadd(boolean inplace, float C) { return new Linear(inplace, 1.0f, C); }
        public Linear ssub(boolean inplace, float C) { return new Linear(inplace, 1.0f, -C); }
        public Linear smul(boolean inplace, float C) { return new Linear(inplace, C, 0.0f); }
        public Linear sdiv(boolean inplace, float C) { return new Linear(inplace, 1.0f / C, 0.0f); }
        public Linear linear(boolean inplace, float alpha, float beta) { return new Linear(inplace, alpha, beta); }

        public Relu relu() { return new Relu(sp_math2_inplace); }
        public Relu relu(boolean inplace) { return new Relu(inplace); }
        
        public ReluN relu6() { return new ReluN(sp_math2_inplace, 6.0f); }
        public ReluN relu6(boolean inplace) { return new ReluN(inplace, 6.0f); }
         
        public ReluN reluN(float N) { return new ReluN(sp_math2_inplace, N); }
        public ReluN reluN(boolean inplace, float N) { return new ReluN(inplace, N); }
        
        public LeakyRelu leakyRelu() { return new LeakyRelu(sp_math2_inplace, leakyRelu_neg_slope); }
        public LeakyRelu leakyRelu(float negative_slope) { return new LeakyRelu(sp_math2_inplace, negative_slope); }
        public LeakyRelu leakyRelu(boolean inplace) { return new LeakyRelu(inplace, leakyRelu_neg_slope); }
        public LeakyRelu leakyRelu(boolean inplace, float negative_slope) { return new LeakyRelu(inplace, negative_slope); }

        public Softplus softplus() { return new Softplus(sp_math2_inplace); }
        public Softplus softplus(boolean inplace) { return new Softplus(inplace); }

        public Elu elu() { return elu(sp_math2_inplace, elu_alpha, elu_neg_slope);}
        public Elu elu(float alpha, float negative_slope) { return new Elu(sp_math2_inplace, alpha, negative_slope); }
        public Elu elu(boolean inplace) { return elu(inplace, elu_alpha, elu_neg_slope); }
        public Elu elu(boolean inplace, float alpha, float negative_slope) { return new Elu(inplace, alpha, negative_slope); }
        
        public Exp exp() { return new Exp(sp_math2_inplace, 1.0f, 0.0f);  }
        public Exp exp(float alpha, float beta) { return new Exp(sp_math2_inplace, alpha, beta); }
        public Exp exp(boolean inplace) { return new Exp(inplace, 1.0f, 0.0f); }
        public Exp exp(boolean inplace, float alpha, float beta) { return new Exp(inplace, alpha, beta); }
        
        public Log log() { return new Log(sp_math2_inplace, 1.0f, 0.0f); }
        public Log log(float alpha, float beta) { return new Log(sp_math2_inplace, alpha, beta); }
        public Log log(boolean inplace) { return new Log(inplace, 1.0f, 0.0f); }
        public Log log(boolean inplace, float alpha, float beta) { return new Log(inplace, alpha, beta); }
        
        public Sqrt sqrt() { return new Sqrt(sp_math2_inplace, 1.0f, 0.0f); }
        public Sqrt sqrt(float alpha, float beta) { return new Sqrt(sp_math2_inplace, alpha, beta); }
        public Sqrt sqrt(boolean inplace) { return new Sqrt(inplace, 1.0f, 0.0f); }
        public Sqrt sqrt(boolean inplace, float alpha, float beta) { return new Sqrt(inplace, alpha, beta); }
        
        public Sigmoid sigmoid() { return new Sigmoid(sp_math2_inplace); }
        public Sigmoid sigmoid(boolean inplace) { return new Sigmoid(inplace); }
        
        public HardSigmoid hard_sigmoid() { return new HardSigmoid(sp_math2_inplace); }
        public HardSigmoid hard_sigmoid(boolean inplace) { return new HardSigmoid(inplace); }
        
        public Tanh tanh() { return new Tanh(sp_math2_inplace); }
        public Tanh tanh(boolean inplace) { return new Tanh(inplace); }

        public Softmax softmax() { return new Softmax(sp_math2_inplace, -1); }
        public Softmax softmax(int features) { return new Softmax(sp_math2_inplace, features);  }
        public Softmax softmax(boolean inplace) { return new Softmax(inplace, -1); }
        public Softmax softmax(boolean inplace, int features) { return new Softmax(inplace, features); }
        
        public LogSoftmax log_softmax() { return new LogSoftmax(sp_math2_inplace, -1); }
        public LogSoftmax log_softmax(int features) { return new LogSoftmax(sp_math2_inplace, features); }
        public LogSoftmax log_softmax(boolean inplace) { return new LogSoftmax(inplace, -1); }
        public LogSoftmax log_softmax(boolean inplace, int features) {  return new LogSoftmax(inplace, features); }

        public HalfSin halfSin(float Amp) { return new HalfSin(sp_math2_inplace, Amp, 1.0f, 0.0f); }
        public HalfSin halfSin(float Amp, float alpha, float beta) {
            return new HalfSin(sp_math2_inplace, Amp, alpha, beta);
        }
        public HalfSin halfSin(boolean inplace, float Amp) {  return halfSin(inplace, Amp, 1.0f, 0.0f); }
        public HalfSin halfSin(boolean inplace, float Amp, float alpha, float beta) {
            return new HalfSin(inplace, Amp, alpha, beta);
        }
          
        public Tan tan() { return new Tan(sp_math2_inplace, 1.0f, 0.0f); }
        public Tan tan(float alpha, float beta) { return new Tan(sp_math2_inplace, alpha, beta); }
        public Tan tan(boolean inplace) { return new Tan(inplace, 1.0f, 0.0f); }
        public Tan tan(boolean inplace, float alpha, float beta) { return new Tan(inplace, alpha, beta); }
        
        public Cot cot() { return new Cot(sp_math2_inplace, 1.0f, 0.0f); }
        public Cot cot(float alpha, float beta) { return new Cot(sp_math2_inplace, alpha, beta); }        
        public Cot cot(boolean inplace) { return new Cot(inplace, 1.0f, 0.0f); }
        public Cot cot(boolean inplace, float alpha, float beta) { return new Cot(inplace, alpha, beta); }

        public Arcsin arcsin() { return new Arcsin(sp_math2_inplace, 1.0f, 0.0f); }
        public Arcsin arcsin(float alpha, float beta) { return new Arcsin(sp_math2_inplace, alpha, beta); }
        public Arcsin arcsin(boolean inplace) { return new Arcsin(inplace, 1.0f, 0.0f); } 
        public Arcsin arcsin(boolean inplace, float alpha, float beta) { return new Arcsin(inplace, alpha, beta); }
        
        public Arctan arctan() { return new Arctan(sp_math2_inplace, 1.0f, 0.0f); } 
        public Arctan arctan(float alpha, float beta) { return new Arctan(sp_math2_inplace, alpha, beta); }
        public Arctan arctan(boolean inplace) { return new Arctan(inplace, 1.0f, 0.0f); } 
        public Arctan arctan(boolean inplace, float alpha, float beta) { return new Arctan(inplace, alpha, beta); } 
        
        public BernouliMul bernouliMul(float p, float v1, float v2) { return new BernouliMul(sp_math2_inplace, p, v1, v2); }
        public BernouliMul bernouliMul(boolean inplace, float p, float v1, float v2) { return new BernouliMul(inplace, p, v1, v2); }
        
        public  Dropout dropout(float nonzero_p) { return new Dropout(sp_math2_inplace, nonzero_p); }
        public  Dropout dropout(boolean inplace, float nonzero_prop) { return new Dropout(inplace, nonzero_prop); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.tensor">
        public View view(int... out_dim) { return new View(sp_tensor_inplace, out_dim); }
        public View view(boolean inplace, int... out_dim) { return new View(inplace, out_dim); }
        
        public Reshape reshape(int... outDim) { return new Reshape(sp_tensor_inplace, outDim); }
        public Reshape reshape(boolean inplace, int... outDim) { return new Reshape(inplace, outDim); }
        
        public Flatten flaten() { return new Flatten(sp_tensor_inplace); }
        public Flatten flaten(boolean inplace) { return new Flatten(inplace); }
        
        public Transpose transpose(int dimIdx1, int dimIdx2) {
            return new Transpose(sp_tensor_inplace, dimIdx1, dimIdx2);
        }
        public Transpose transpose(boolean inplace, int dimIdx1, int dimIdx2) {
            return new Transpose(inplace, dimIdx1, dimIdx2);
        }
        
        public Rot180 rot180() { return new Rot180(sp_tensor_inplace); }
        public Rot180 rot180(boolean inplace) { return new Rot180(inplace); }
        
        public Pad pad(int... p) { return new Pad(sp_tensor_inplace, p, Vector.arrayCopy(p)); }
        public Pad pad(int[] p0, int[] p1) { return new Pad(sp_tensor_inplace, p0, p1); }
        public Pad pad(boolean inplace, int... p) { return new Pad(inplace, p, Vector.arrayCopy(p)); }
        public Pad pad(boolean inplace, int[] p0, int[] p1) { return new Pad(inplace, p0, p1); }
        
        public Trim trim(int... t) { return new Trim(sp_tensor_inplace, t, Vector.arrayCopy(t)); }
        public Trim trim(int[] t0, int[] t1) { return new Trim(sp_tensor_inplace, t0, t1); }
        public Trim trim(boolean inplace, int... t) {  return new Trim(inplace, t, Vector.arrayCopy(t)); }
        public Trim trim(boolean inplace, int[] t0, int[] t1) { return new Trim(inplace, t0, t1); }
        
        public Pad pad2D(int... p) { 
            return new Pad(sp_tensor_inplace, Vector.append(p, 0), Vector.append(p, 0)); 
        }
        public Pad pad2D(int[] p0, int[] p1) {
            return new Pad(sp_tensor_inplace, Vector.append(p0, 0), Vector.append(p1, 0)); 
        }
        public Pad pad2D(boolean inplace, int... p) { 
            return new Pad(inplace, Vector.append(p, 0), Vector.append(p, 0)); 
        }
        public Pad pad2D(boolean inplace, int[] p0, int[] p1) { 
            return new Pad(inplace, Vector.append(p0, 0), Vector.append(p1, 0)); 
        }
        
        public Trim trim2D(int... t) {
            return new Trim(sp_tensor_inplace, Vector.append(t, 0), Vector.append(t, 0)); 
        }
        public Trim trim2D(int[] t0, int[] t1) {
            return new Trim(sp_tensor_inplace, Vector.append(t0, 0), Vector.append(t1, 0));
        }
        public Trim trim2D(boolean inplace, int... t) {  
            return new Trim(inplace, Vector.append(t, 0), Vector.append(t, 0)); 
        }
        public Trim trim2D(boolean inplace, int[] t0, int[] t1) { 
            return new Trim(inplace, Vector.append(t0, 0), Vector.append(t1, 0)); 
        }
        
        public Expand expand(int...out_dim) { return new Expand(sp_tensor_inplace, Engine.from_center, out_dim); }
        public Expand expand(int[] start_point, int[] out_dim) { return new Expand(sp_tensor_inplace, start_point, out_dim); }
        public Expand expand(boolean inplace, int...out_dim) { return new Expand(inplace, Engine.from_center, out_dim); }
        public Expand expand(boolean inplace, int[] start_point, int[] out_dim) { return new Expand(inplace, start_point, out_dim); }
        
        public Crop crop(int...out_dim) { return new Crop(sp_tensor_inplace, Engine.from_center, out_dim); }
        public Crop crop(int[] start_point, int[] out_dim) { return new Crop(sp_tensor_inplace, start_point, out_dim); }
        public Crop crop(boolean inplace, int...out_dim) { return new Crop(inplace, Engine.from_center, out_dim); }
        public Crop crop(boolean inplace, int[] start_point, int[] out_dim) { return new Crop(inplace, start_point, out_dim); }
        
        public Expand expand2D(int...out_dim) {
            return new Expand(sp_tensor_inplace, Engine.from_center, Vector.append(out_dim, -1));
        }
        public Expand expand2D(int[] start, int[] out_dim) {
            return new Expand(sp_tensor_inplace, Vector.append(start, 0), Vector.append(out_dim, -1));
        }
        public Expand expand2D(boolean inplace, int...out_dim) {
            return new Expand(inplace, Engine.from_center, Vector.append(out_dim, -1));
        }
        public Expand expand2D(boolean inplace, int[] start, int[] out_dim) {
            return new Expand(inplace, Vector.append(start, 0), Vector.append(out_dim, -1));
        }
        
        public Crop crop2D(int...out_dim) {
            return new Crop(sp_tensor_inplace, Engine.from_center, Vector.append(out_dim, -1));
        }
        public Crop crop2D(int[] start, int[] out_dim) {
            return new Crop(sp_tensor_inplace, Vector.append(start, 0), Vector.append(out_dim, -1));
        }
        public Crop crop2D(boolean inplace, int...out_dim) { 
            return new Crop(inplace, Engine.from_center, Vector.append(out_dim, -1));
        }
        public Crop crop2D(boolean inplace, int[] start, int[] out_dim) {
            return new Crop(inplace, Vector.append(start, 0), Vector.append(out_dim, -1));
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.affine">
        public Affine affine(int... feature_dim) { return new Affine(sp_affine_inplace, feature_dim); }
        public Affine affine(boolean inplace, int... feature_dim) { return new Affine(inplace, feature_dim); }
        
        public LayerNorm layerNorm(int... feature_dim) {
            return new LayerNorm(sp_affine_inplace, layerNorm_affine, layerNorm_eps, feature_dim);
        }     
        public LayerNorm layerNorm(boolean affine, float eps, int... feature_dim) {
            return new LayerNorm(sp_affine_inplace, affine, eps, feature_dim);
        }
        public LayerNorm layerNorm(boolean inplace, int... feature_dim) {
            return new LayerNorm(inplace, layerNorm_affine, layerNorm_eps, feature_dim);
        }     
        public LayerNorm layerNorm(boolean inplace, boolean affine, float eps, int... feature_dim) {
            return new LayerNorm(inplace, affine, eps, feature_dim);
        }

        public GlobalSqBatchNorm global_sqBatchNorm(int... feature_dim) {
            return new GlobalSqBatchNorm(sp_affine_inplace, batchNorm_affine, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalSqBatchNorm(sp_affine_inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean inplace, int... feature_dim) {
            return new GlobalSqBatchNorm(inplace, true, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalSqBatchNorm global_sqBatchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalSqBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        public SqBatchNorm sqBatchNorm(int... feature_dim) {
            return new SqBatchNorm(sp_affine_inplace, batchNorm_affine, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new SqBatchNorm(sp_affine_inplace, batchNorm_affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean inplace, int... feature_dim) {
            return new SqBatchNorm(inplace, true, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public SqBatchNorm sqBatchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new SqBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        public GlobalBatchNorm global_batchNorm(int... feature_dim) {
            return new GlobalBatchNorm(sp_affine_inplace, batchNorm_affine,
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalBatchNorm(sp_affine_inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean inplace, int... feature_dim) {
            return new GlobalBatchNorm(inplace, true, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public GlobalBatchNorm global_batchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new GlobalBatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        public BatchNorm batchNorm(int... feature_dim) {
            return new BatchNorm(sp_affine_inplace, batchNorm_affine, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public BatchNorm batchNorm(boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new BatchNorm(sp_affine_inplace, batchNorm_affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        
        public BatchNorm batchNorm(boolean inplace, int... feature_dim) {
            return new BatchNorm(inplace, true, 
                    batchNorm_beta1, batchNorm_beta2, batchNorm_eps,
                    feature_dim);
        }
        public BatchNorm batchNorm(boolean inplace,  boolean affine,
                float beta1, float beta2, float eps, 
                int... feature_dim) {
            return new BatchNorm(inplace, affine, 
                    beta1, beta2, eps,
                    feature_dim);
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.AvgPool2D">
        public AvgPool2D avgPool2D(int div) {
            return new AvgPool2D(avgpool2D_igpad,
                    div, div, div, div, 0, 0, 
                    -1, -1);
        }
        public AvgPool2D avgPool2D(int kernel, int stride, int padding) {
            return new AvgPool2D(avgpool2D_igpad,
                    kernel, kernel, stride, stride, padding, padding, 
                    -1, -1);
        }
        public AvgPool2D avgPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new AvgPool2D(avgpool2D_igpad,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width, 
                    -1, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.MaxPool2D">
        public MaxPool2D maxPool2D(int div) { return new MaxPool2D(div, div, div, div, 0, 0, -1, -1); }
        public MaxPool2D maxPool2D(int kernel, int stride, int padding) {
            return new MaxPool2D(kernel, kernel, stride, stride, padding, padding, 
                    -1, -1);
        }
        public MaxPool2D maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width) {
            return new MaxPool2D(
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    -1, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.AvgUnpool2D">
        public AvgUnpool2D avgUnpool2D(int mul) {
            return new AvgUnpool2D(avgunpool2D_ignore_padding, 
                    mul, mul, mul, mul, 0, 0, 
                    -1, -1);
        }
        public AvgUnpool2D avgUnpool2D(int kernel, int stride, int padding) {
            return new AvgUnpool2D(avgunpool2D_ignore_padding, 
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1);
        }
        public AvgUnpool2D avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width)
        {
            return new AvgUnpool2D(avgunpool2D_ignore_padding,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width,
                    -1, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.AdaptivePool2D">
        public AdaptiveAvgPool2D adaptive_avgPool2D(int out_size) { return new AdaptiveAvgPool2D(avgpool2D_igpad, out_size, out_size); }
        public AdaptiveAvgPool2D adaptive_avgPool2D(int out_height, int out_width) {
            return new AdaptiveAvgPool2D(avgpool2D_igpad, out_height, out_width);
        }
        
        public AdaptiveMaxPool2D adaptive_maxPool2D(int out_size) { return new AdaptiveMaxPool2D(out_size, out_size); }
        public AdaptiveMaxPool2D adaptive_maxPool2D(int out_height, int out_width) {
            return new AdaptiveMaxPool2D(out_height, out_width);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.Pool1D">
        public AvgPool1D avgPool1D(int div) { return new AvgPool1D(avgpool1D_igpad, div, div, 0, -1); }
        public AvgPool1D avgPool1D(int kernel_width, int stride_width, int padding_width) {
            return new AvgPool1D(avgpool1D_igpad, kernel_width, stride_width, padding_width, -1);
        }
        
        public MaxPool1D maxPool1D(int div) { return new MaxPool1D(div, div, 0, -1); }
        public MaxPool1D maxPool1D(int kernel_width, int stride_width, int padding_width) {
            return new MaxPool1D(kernel_width, stride_width, padding_width, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.pool.AdaptivePool1D">
        public AdaptiveAvgPool1D adaptive_avgPool1D(int out_width) { return new AdaptiveAvgPool1D(avgpool1D_igpad, out_width); }
        public AdaptiveMaxPool1D adaptive_maxPool1D(int out_width) { return new AdaptiveMaxPool1D(out_width); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: simple.blas.Conv3D">
        public Conv3D point_conv3D(boolean biased, 
                int in_channel, int out_channel) {
            return new Conv3D(biased, 
                    in_channel, out_channel,
                    1, 1, 1, 1, 0, 0, 
                    -1, -1);
        }
        
        public Conv3D conv3D(boolean biased,
                int in_channel, int out_channel,
                int kernel, int stride, int padding) {
            return new Conv3D(biased, 
                    in_channel, out_channel,
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1);
        }
        
        public Conv3D conv3D(boolean biased,
                int in_channel,     int out_channel,
                int kernel_height,  int kernel_width,
                int stride_height,  int stride_width,
                int padding_height, int padding_width) {
            return new Conv3D(biased, 
                    in_channel,     out_channel,
                    kernel_height,  kernel_width,
                    stride_height,  stride_width,
                    padding_height, padding_width,
                    -1, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.blas.Deconv3D">
        public Deconv3D point_deconv3D(boolean biased, 
                int in_channel, int out_channel) {
            return new Deconv3D(biased, 
                    in_channel, out_channel, 
                    1, 1, 1, 1, 0, 0, 
                    -1, -1);
        }
        public Deconv3D deconv3D(boolean biased,
                int in_channel, int out_channel,
                int kernel, int stride, int padding) {
            return new Deconv3D(biased, 
                    in_channel, out_channel,
                    kernel, kernel,
                    stride, stride,
                    padding, padding,
                    -1, -1);
        }
        public Deconv3D deconv3D(boolean biased, 
                int in_channel,     int out_channel,
                int kernel_height,  int kernel_width,
                int stride_height,  int stride_width,
                int padding_height, int padding_width) {
            return new Deconv3D(biased, 
                    in_channel,     out_channel,
                    kernel_height,  kernel_width,
                    stride_height,  stride_width,
                    padding_height, padding_width,
                    -1, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.blas.Conv2D">
        public Conv2D point_conv2D(boolean biased, 
                int in_channel, int out_channel) {
            return new Conv2D(biased, 
                    in_channel, out_channel,
                    1, 1, 0, -1);
        }
        
        public Conv2D conv2D(boolean biased,
                int in_channel, int out_channel,
                int kernel_width, int stride_width, int padding_width) {
            return new Conv2D(biased, 
                    in_channel, out_channel,
                    kernel_width, stride_width, padding_width, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.blas.Deconv2D">
        public Deconv2D point_deconv2D(boolean biased, 
                int in_channel, int out_channel) {
            return new Deconv2D(biased, 
                    in_channel, out_channel, 
                    1, 1, 0, -1);
        }
        public Deconv2D deconv2D(boolean biased,
                int in_channel, int out_channel,
                int kernel_width, int stride_width, int padding_width) {
            return new Deconv2D(biased, 
                    in_channel, out_channel,
                    kernel_width, stride_width, padding_width, -1);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: simple.blas.Fullconnect">
        public FullConnect fullconnect(boolean biased, int in_features, int out_features) {
            return new FullConnect(biased, in_features, out_features);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: dual.bias">
        public MatMul matMul() { return new MatMul(); }
        public MatMulT1 matMulT1() { return new MatMulT1(); }
        public MatMulT2 matMulT2() { return new MatMulT2(); }
        
        public BatchMatMul batchMatMul() {  return batchMatMul(dl_likeX1); }
        public BatchMatMul batchMatMul(boolean likeX1) { return new BatchMatMul(likeX1); }
        
        public BatchMatMulT1 batchMatMulT1() { return batchMatMulT1(dl_likeX1);  }
        public BatchMatMulT1 batchMatMulT1(boolean likeX1) { return new BatchMatMulT1(likeX1); }
        
        public BatchMatMulT2 batchMatMulT2() { return batchMatMulT2(dl_likeX1); }
        public BatchMatMulT2 batchMatMulT2(boolean likeX1) { return new BatchMatMulT2(likeX1); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: dual.math">
        public Linear2 add() { return new Linear2(dl_likeX1, 1.0f,  1.0f, 0.0f); }
        public Linear2 sub() { return new Linear2(dl_likeX1, 1.0f, -1.0f, 0.0f); }
        public Linear2 add(float alpha, float beta) { return new Linear2(dl_likeX1, alpha,  beta, 0.0f); }
        public Linear2 sub(float alpha, float beta) { return new Linear2(dl_likeX1, alpha, -beta, 0.0f); } 
        public Linear2 linear2(float alpha, float beta, float gamma) {
            return new Linear2(dl_likeX1, alpha, beta, gamma);
        }
        
        public Linear2 add(boolean likeX1) { return new Linear2(likeX1, 1.0f, 1.0f, 0.0f); }
        public Linear2 sub(boolean likeX1) { return new Linear2(likeX1, 1.0f, -1.0f, 0.0f); }
        public Linear2 add(boolean likeX1, float alpha, float beta) { return new Linear2(likeX1, alpha, beta, 0.0f); }
        public Linear2 sub(boolean likeX1, float alpha, float beta) { return new Linear2(likeX1, alpha, -beta, 0.0f); }
        public Linear2 linear2(boolean likeX1, float alpha, float beta, float gamma) {
            return new Linear2(likeX1, alpha, beta, gamma);
        }
        
        public Linear2Row add_row() { return new Linear2Row(1.0f,  1.0f, 0.0f); }
        public Linear2Row sub_row() { return new Linear2Row(1.0f, -1.0f, 0.0f); }
        public Linear2Row add_row(float alpha, float beta) { return new Linear2Row(alpha,  beta, 0.0f); }
        public Linear2Row sub_row(float alpha, float beta) { return new Linear2Row(alpha, -beta, 0.0f); } 
        public Linear2Row linear2_row(float alpha, float beta, float gamma) {
            return new Linear2Row(alpha, beta, gamma);
        }
        
        public Quadratic2 mul() { return new Quadratic2(dl_likeX1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2 mul(float alpha) { return new Quadratic2(dl_likeX1, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2 sqadd() { return new Quadratic2(dl_likeX1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2 sqsub() { return new Quadratic2(dl_likeX1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2 sqadd(float alpha, float beta) { return new Quadratic2(dl_likeX1, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2 quadratic2(float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2(dl_likeX1, k11, k12, k22, k1, k2, C);
        }
        
        public Quadratic2 mul(boolean likeX1) { return new Quadratic2(likeX1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2 mul(boolean likeX1, float alpha) { return new Quadratic2(likeX1, 0.0f, alpha, 0.0f,  0.0f, 0.0f, 0.0f); }
        public Quadratic2 sqadd(boolean likeX1) { return new Quadratic2(likeX1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f); }       
        public Quadratic2 sqsub(boolean likeX1) { return new Quadratic2(likeX1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f); }      
        public Quadratic2 sqadd(boolean likeX1, float alpha, float beta) { return new Quadratic2(likeX1, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2 quadratic2(boolean likeX1, float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2(likeX1, k11, k12, k22, 
                    k1, k2, C);
        }
        
        public Quadratic2Row mul_row() {return new Quadratic2Row(0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2Row mul_row(float alpha) { return new Quadratic2Row(0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2Row sqadd_row() { return new Quadratic2Row(1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Row sqsub_row() { return new Quadratic2Row(1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Row sqadd_row(float alpha, float beta) { return new Quadratic2Row(alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Row quadratic2_row(float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2Row(k11, k12, k22, k1, k2, C);
        }
        
        public Quadratic2Center mul_center() {return new Quadratic2Center(-1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2Center mul_center(float alpha) { return new Quadratic2Center(-1, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f); }
        public Quadratic2Center sqadd_center() { return new Quadratic2Center(-1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Center sqsub_center() { return new Quadratic2Center(-1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Center sqadd_center(float alpha, float beta) { return new Quadratic2Center(-1, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f); }        
        public Quadratic2Center quadratic2_center(float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2Center(-1, k11, k12, k22, k1, k2, C);
        }
        public Quadratic2Center quadratic2_center(int dim2, float k11, float k12, float k22, float k1, float k2, float C) {
            return new Quadratic2Center(dim2, k11, k12, k22, k1, k2, C);
        }
        
        public Div div() { return div(dl_likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); }
        public Div div(float alpha1, float beta1, float alpha2, float beta2, float gamma){
            return div(dl_likeX1, alpha1, beta1, alpha2, beta2, gamma);
        }
        
        public Div div(boolean likeX1) { return div(likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); }
        public Div div(boolean likeX1, float alpha1, float beta1, float alpha2, float beta2, float gamma){
            return new Div(likeX1, alpha1, beta1, alpha2, beta2, gamma);
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="create: reducer.math">
        public LinearSummary sum() { return new LinearSummary(1.0f, 0.0f);  }
        public LinearSummary sum(float alpha) { return new LinearSummary(alpha, 0.0f); }
        public LinearSummary linear_sum(float alpha, float beta) {
            return new LinearSummary(alpha, beta); 
        }
        
        public LinearMean mean() { return new LinearMean(1.0f, 0.0f); }
        public LinearMean mean(float alpha) { return new LinearMean(alpha, 0.0f); }
        public LinearMean linearMean(float alpha, float beta) { 
            return new LinearMean(alpha, beta); 
        }
        
        public QuadraticSummary sqsum() { return new QuadraticSummary(1.0f, 0.0f, 0.0f); }
        public QuadraticSummary sqsum(float alpha) { return new QuadraticSummary(alpha, 0.0f, 0.0f); }
        public QuadraticSummary quadratic_sum(float alpha, float beta, float gamma) {
            return new QuadraticSummary(alpha, beta, gamma);
        }
        
        public QuadraticMean sqmean() { return new QuadraticMean(1.0f, 0.0f, 0.0f); }
        public QuadraticMean sqmean(float alpha) { return new QuadraticMean(alpha, 0.0f, 0.0f); }
        public QuadraticMean quadratic_mean(float alpha, float beta, float gamma) {
            return new QuadraticMean(alpha, beta, gamma);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="create: reducer.tensor">
        public Concat concat() { return new Concat(-1); }
        public Concat concat(int dimIdx) { return new Concat(dimIdx); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: furcation">
        public Split split(int dimIdx, int... section) { return new Split(dimIdx, section); }
        
        public Chunk chunk(int n) { return new Chunk(-1, n); }
        public Chunk chunk(int dimIdx, int n) { return new Chunk(dimIdx, n); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: complex">
        public Sequence sequence(Collection<Unit> units) { return new Sequence(units); }
        public Sequence sequence(Unit... units) { return new Sequence(units); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="utils">
        public void print_param_stat(Unit unit) { 
            TreeMap<String, Parameter> map = new TreeMap<>(unit.param_map());
            map.forEach((String name, Parameter p) -> {
                System.out.print("name = [" + name + " ]");
                Tensor tensor = p.ts();
                System.out.print("\t, mean = [ " + tensor.mean().get() + " ]");
                System.out.print("\t, std = [" + tensor.std().get() + "]\n");
            });
        }
        
        public void constant_params(Unit unit, float value) {
            for(Parameter p : unit.params()) p.ts().constant(value).c();
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: UnitFunctional">
    public static class UnitFunctional {
        protected UnitFunctional() {}
        
        public static final UnitFunctional F = new UnitFunctional();
        
        //<editor-fold defaultstate="collapsed" desc="axuilary: single-instance units">
        private static final SimpleFunction sp_func = new SimpleFunction() {//simple function
            { name = "Alpha.functional.simple_function"; }
            private static final long serialVersionUID = 1L;
            @Override public final boolean need_grads() { return false; }//without params
            @Override public final void append(String pre, StringBuilder sb) { sb.append(name); }
            @Override protected final SimpleCore<?> create_unit_core() { return null; }
            @Override protected UnitCoreManager create_ucm() { return null; } 
            @Override public int index_of_core(UnitCore core) { return -1; }
        };
        
        private static final DualFunction du_func = new DualFunction() {//dual function
            { name = "Alpha.functional.dual_function"; }
            private static final long serialVersionUID = 1L;
            @Override public final boolean need_grads() { return false; }//without params
            @Override public final void append(String pre, StringBuilder sb) { sb.append(name); }
            @Override protected final DualCore<?> create_unit_core() { return null; }
            @Override protected UnitCoreManager create_ucm() { return null; } 
            @Override public int index_of_core(UnitCore core) { return -1; }
        };
        
        private static final ReduceFunction re_func = new ReduceFunction() {
            { name = "Alpha.functional.reduce_function"; }
            private static final long serialVersionUID = 1L;
            @Override public final boolean need_grads() { return false; }//without params
            @Override public final void append(String pre, StringBuilder sb) { sb.append(name); }
            @Override protected ReducerCore<?> create_unit_core() { return null; }
            @Override protected UnitCoreManager create_ucm() { return null; } 
            @Override public int index_of_core(UnitCore core) { return -1; }
        }; 
        
        private static final FurcateFunction fc_func = new FurcateFunction() {
            { name = "Alpha.functional.furcate_function"; }
            private static final long serialVersionUID = 1L;
            @Override public final boolean need_grads() { return false; }//without params
            @Override public final void append(String pre, StringBuilder sb) { sb.append(name); }
            @Override protected FurcationCore<?> create_unit_core() { return null; }
            @Override protected UnitCoreManager create_ucm() { return null; } 
            @Override public int index_of_core(UnitCore core) { return -1; }
        };
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="auxilary: carrier">
        //<editor-fold defaultstate="collapsed" desc="carrier: simple">
        private static Tensor[] csp(SimpleCore<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            //if X is output of OneOffUnitCore, X.need_carry = true 
            Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX().need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        
        private static Tensor[] csp(boolean inplace, SimpleCore<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            if(!inplace) {//if X[0] is output of OneOffScale, X[0].needCarry = true
                Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
                core.hook_after_backward((self)->{
                    Tensor deltaX = core.deltaX().need_carry(true);
                    deltaX.carry(core.deltaY());
                });
            }
            return Y;
        }
        
        private static Tensor[] fsp(SimpleCore<?> core, Tensor[] X) {//simple fixed-carrier
            Tensor[] Y = core.forward(X);
            if(!sp_math2_inplace){//if X[0] is output of OneOffScale, X[0].needCarry = true
                Y[0].need_carry(true); Y[0].carry(X[0]);//use Y[0] to carray X[0]
                core.hook_after_backward((self)->{
                    Tensor deltaX = core.deltaX().need_carry(true);
                    deltaX.carry(core.deltaY());
                });
            }
            return Y;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="carraier: simple.MaxPool">
        private static Tensor[] csp_maxPool(CoreMaxPool1D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Idx(CoreMaxPool1D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return new Tensor[] { Y[0], core.Index() };
        }
        
        private static Tensor[] csp_maxPool(CoreMaxPool2D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Idx(CoreMaxPool2D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return new Tensor[] { Y[0], core.Index() };
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="carraier: simple.AdaptiveMaxPool">
        private static Tensor[] csp_maxPool(CoreAdaptiveMaxPool1D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Idx(CoreAdaptiveMaxPool1D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return new Tensor[] {Y[0], core.Index()};
        }
        
        private static Tensor[] csp_maxPool(CoreAdaptiveMaxPool2D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index); //use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        
        private static Tensor[] csp_maxPool_with_Idx(CoreAdaptiveMaxPool2D<?> core, Tensor[] X) {//simple carrier
            Tensor[] Y = core.forward(X);
            Y[0].need_carry(true);//if X[0] is output of OneOffUnitCore, X[0].need_carry = true 
            Tensor Index = core.Index(); Index.need_carry(true);
            Y[0].carry(X[0]); Y[0].carry(Index);//use Y[0] to carray { X[0], Index }
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return new Tensor[] {Y[0], core.Index()};
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="carrier: dual">
        private static Tensor[] cdu(DualCore<?> core, Tensor[] X) {//dual scale carrier
            Tensor[] Y = core.forward(X);//if X is output of OneOffUnitCore, X.need_carry = true 
            Y[0].need_carry(true); Y[0].carry(X);//X1, X2 -> Y
            core.hook_after_backward((self)->{//deltaY -> deltaX1, deltaX2
                Tensor deltaX1 = core.deltaX1(); deltaX1.need_carry(true);
                Tensor deltaX2 = core.deltaX2(); deltaX2.need_carry(true);
                deltaX1.carry(core.deltaY());
            });
            return Y;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="carrier: reducer">
        private static Tensor[] cre(ReducerCore<?> core, Tensor[] X) {//Reducer Carrier
            Tensor[] Y = core.forward(X);//if X is output of OneOffUnitCore, X.need_carry = true
            Y[0].need_carry(true); Y[0].carry(X);//X[0: N-1] -> Y
            core.hook_after_backward((self)->{
                Tensor[] deltaX = core.deltaX();
                for(Tensor dX : deltaX) dX.need_carry(true);
                deltaX[0].carry(core.deltaY());
            });
            return Y;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="carrier: furcation">
        private Tensor[] cfc(FurcationCore<?> core, Tensor[] X) {//one input, multi output
            Tensor[] Y = core.forward(X);//if X is output of OneOffUnitCore, X.need_carry = true
            for (Tensor out : Y) out.need_carry(true);
            Y[0].carry(X[0]);
            core.hook_after_backward((self)->{
                Tensor deltaX = core.deltaX(); deltaX.need_carry(true);
                deltaX.carry(core.deltaY());
            });
            return Y;
        }
        //</editor-fold>
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.math1">
        public final Tensor[] abs(Tensor... X) {  return csp(new CoreAbs<>(sp_func, 1.0f, 0.0f), X); }
        public final Tensor[] abs(float alpha, float beta, Tensor...X) { return csp(new CoreAbs<>(sp_func, alpha, beta), X); }
        
        public final Tensor[] square(Tensor... X) { return csp(new CoreQuadratic<>(sp_func, 1.0f, 1.0f, 0.0f), X); }
        public final Tensor[] square(float alpha, Tensor... X) { return csp(new CoreQuadratic<>(sp_func, alpha, 0.0f, 0.0f), X); }
        public final Tensor[] quadratic(float alpha, float beta, float gamma, Tensor... X){ return csp(new CoreQuadratic<>(sp_func, alpha, beta, gamma), X); }
        
        public final Tensor[] sin(Tensor... X) { return csp(new CoreSin<>(sp_func, 1.0f, 0), X); }
        public final Tensor[] sin(float alpha, float beta, Tensor... X) { return csp(new CoreSin<>(sp_func, alpha, beta), X); }
        
        public final Tensor[] cos(Tensor... X) { return csp(new CoreCos<>(sp_func, 1.0f, 0), X); }
        public final Tensor[] cos(float alpha, float beta, Tensor... X) { return csp(new CoreCos<>(sp_func, alpha, beta), X); }
        
        public final Tensor[] csc(Tensor... X) { return csp(new CoreCsc<>(sp_func, 1.0f, 0.0f), X); }
        public final Tensor[] csc(float alpha, float beta, Tensor... X) { return csp(new CoreCsc<>(sp_func, alpha, beta), X); }
        
        public final Tensor[] sec(Tensor... X) { return csp(new CoreSec<>(sp_func, 1.0f, 0.0f), X); }
        public final Tensor[] sec(float alpha, float beta, Tensor... X) { return csp(new CoreSec<>(sp_func, alpha, beta), X); }
        
        public final Tensor[] gelu(Tensor... X) { return csp(new CoreGelu<>(sp_func), X); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.math2">
        public final Tensor[] max(float vmax, Tensor... X) { return fsp(new CoreMax<>(sp_func, sp_math2_inplace, 1.0f, 0.0f, vmax), X); }
        public final Tensor[] max(float alpha, float beta, float vmax, Tensor... X) {
            return fsp(new CoreMax<>(sp_func, sp_math2_inplace, alpha, beta, vmax), X);
        }
        public final Tensor[] max(boolean inplace, float vmax, Tensor... X) { return csp(inplace, new CoreMax<>(sp_func, inplace, 1.0f, 0.0f, vmax), X); }  
        public final Tensor[] max(boolean inplace, float alpha, float beta, float vmax, Tensor... X) {
            return csp(inplace, new CoreMax<>(sp_func, inplace, alpha, beta, vmax), X);
        }
        
        public final Tensor[] min(float vmin, Tensor...X) { return fsp(new CoreMin<>(sp_func, sp_math2_inplace, 1.0f, 0.0f, vmin), X); }
        public final Tensor[] min(float alpha, float beta, float vmin, Tensor... X) {
            return fsp(new CoreMin<>(sp_func, sp_math2_inplace, alpha, beta, vmin), X);
        }
        public final Tensor[] min(boolean inplace, float vmin, Tensor... X) { return csp(inplace, new CoreMin<>(sp_func, inplace, 1.0f, 0.0f, vmin), X); }
        public final Tensor[] min(boolean inplace, float alpha, float beta, float vmin, Tensor... X) {
            return csp(inplace, new CoreMin<>(sp_func, inplace, alpha, beta, vmin), X);
        }
        
        public final Tensor[] clip(float vmin, float vmax, Tensor... X) { return fsp(new CoreClip<>(sp_func, sp_math2_inplace, 1.0f, 0.0f, vmin, vmax), X); }
        public final Tensor[] clip(float alpha, float beta, float vmin, float vmax, Tensor... X) {
            return fsp(new CoreClip<>(sp_func, sp_math2_inplace, alpha, beta, vmin, vmax), X);
        }
        public final Tensor[] clip(boolean inplace, float vmin, float vmax, Tensor... X) { return csp(inplace, new CoreClip<>(sp_func, inplace, 1.0f, 0.0f, vmin, vmax), X); }
        public final Tensor[] clip(boolean inplace, float alpha, float beta, float vmin, float vmax, Tensor... X) {
            return csp(inplace, new CoreClip<>(sp_func, inplace, alpha, beta, vmin, vmax), X);
        }
        
        public final Tensor[] rpl(Tensor... X) { return fsp(new CoreRpl<>(sp_func, sp_math2_inplace, 1.0f, 0.0f, 0.0f), X); }
        public final Tensor[] rpl(float alpha, float beta, float gamma, Tensor... X) {
            return fsp(new CoreRpl<>(sp_func, sp_math2_inplace, alpha, beta, gamma), X);
        }
        public final Tensor[] rpl(boolean inplace, Tensor... X) { return csp(inplace, new CoreRpl<>(sp_func, inplace, 1.0f, 0.0f, 0.0f), X); }
        public final Tensor[] rpl(boolean inplace, float alpha, float beta, float gamma, Tensor... X) {
            return csp(inplace, new CoreRpl<>(sp_func, inplace, alpha, beta, gamma), X);
        }
        
        public final Tensor[] sadd(float C, Tensor... X) { return fsp(new CoreLinear<>(sp_func, sp_math2_inplace, 1.0f, C), X);  }
        public final Tensor[] ssub(float C, Tensor... X) { return fsp(new CoreLinear<>(sp_func, sp_math2_inplace, 1.0f, -C), X); }
        public final Tensor[] smul(float C, Tensor... X) { return fsp(new CoreLinear<>(sp_func, sp_math2_inplace,  C, 0.0f), X); }
        public final Tensor[] sdiv(float C, Tensor... X) { return fsp(new CoreLinear<>(sp_func, sp_math2_inplace, (1.0f / C), 0.0f), X);  }
        public final Tensor[] linear(float alpha, float beta, Tensor... X) {
            return fsp(new CoreLinear<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] sadd(boolean inplace, float C, Tensor... X){ return csp(inplace, new CoreLinear<>(sp_func, inplace, 1.0f, C), X); }
        public final Tensor[] ssub(boolean inplace, float C, Tensor... X){ return csp(inplace, new CoreLinear<>(sp_func, inplace, 1.0f, -C), X); }
        public final Tensor[] smul(boolean inplace, float C, Tensor... X){ return csp(inplace, new CoreLinear<>(sp_func, inplace, C, 0.0f), X); }
        public final Tensor[] sdiv(boolean inplace, float C, Tensor... X){ return csp(inplace, new CoreLinear<>(sp_func, inplace, (1.0f / C), 0.0f), X); }
        public final Tensor[] linear(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, new CoreLinear<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] exp(Tensor... X) { return fsp(new CoreExp<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] exp(float alpha, float beta, Tensor... X) { 
            return fsp(new CoreExp<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] exp(boolean inplace, Tensor... X) { return csp(inplace, new CoreExp<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] exp(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, new CoreExp<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] log(Tensor... X) { return fsp(new CoreLog<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] log(float alpha, float beta, Tensor... X) { 
            return fsp(new CoreLog<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] log(boolean inplace, Tensor... X) { return csp(inplace, new CoreLog<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] log(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, new CoreLog<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] sqrt(Tensor... X) { return fsp(new CoreSqrt<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] sqrt(float alpha, float beta, Tensor... X) { 
            return fsp(new CoreSqrt<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] sqrt(boolean inplace,  Tensor... X) { return csp(inplace, new CoreSqrt<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] sqrt(boolean inplace, float alpha, float beta, Tensor... X) {
            return csp(inplace, new CoreSqrt<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] relu(Tensor... X) { return fsp(new CoreRelu<>(sp_func, sp_math2_inplace), X); }
        public final Tensor[] relu(boolean inplace, Tensor... X) { return csp(inplace, new CoreRelu<>(sp_func, inplace), X); }
        
        public final Tensor[] relu6(Tensor... X) { return fsp(new CoreReluN<>(sp_func, sp_math2_inplace, 6.0f), X); }
        public final Tensor[] relu6(boolean inplace, Tensor... X) { return csp(inplace, new CoreReluN<>(sp_func, inplace, 6.0f), X); }
        
        public final Tensor[] reluN(float N, Tensor... X) { return fsp(new CoreReluN<>(sp_func, sp_math2_inplace, N), X); }
        public final Tensor[] reluN(boolean inplace, float N, Tensor... X) { return csp(inplace, new CoreReluN<>(sp_func, inplace, N), X); }
        
        public final Tensor[] leakyRelu(Tensor... X) { return fsp(new CoreLeakyRelu<>(sp_func, sp_math2_inplace, leakyRelu_neg_slope), X); }
        public final Tensor[] leakyRelu(float negative_slope, Tensor... X) {
            return fsp(new CoreLeakyRelu<>(sp_func, sp_math2_inplace, negative_slope), X);
        }
        public final Tensor[] leakyRelu(boolean inplace, Tensor... X) { return csp(inplace, new CoreLeakyRelu<>(sp_func, inplace, leakyRelu_neg_slope), X);}
        public final Tensor[] leakyRelu(boolean inplace, float negative_slope, Tensor... X) {
            return csp(inplace, new CoreLeakyRelu<>(sp_func, inplace, negative_slope), X);
        }

        public final Tensor[] softplus(Tensor... X) { return fsp(new CoreSoftplus<>(sp_func, sp_math2_inplace), X); }
        public final Tensor[] softplus(boolean inplace, Tensor... X) { return csp(inplace, new CoreSoftplus<>(sp_func, inplace), X); }

        public final Tensor[] elu(Tensor... X) { return fsp(new CoreElu<>(sp_func, sp_math2_inplace, elu_alpha, elu_neg_slope), X); }
        public final Tensor[] elu(float alpha, float negative_slope, Tensor... X) {
            return fsp(new CoreElu<>(sp_func, sp_math2_inplace, alpha, negative_slope), X);
        }
        public final Tensor[] elu(boolean inplace, Tensor... X) { return csp(inplace, new CoreElu<>(sp_func, inplace, elu_alpha, elu_neg_slope), X); }
        public final Tensor[] elu(boolean inplace, float alpha, float negative_slope, Tensor... X) {
            return csp(inplace, new CoreElu<>(sp_func, inplace, alpha, negative_slope), X);
        }
        
        public final Tensor[] sigmoid(Tensor... X) { return fsp(new CoreSigmoid<>(sp_func, sp_math2_inplace), X); }
        public final Tensor[] sigmoid(boolean inplace, Tensor... X) { return csp(inplace, new CoreSigmoid<>(sp_func, inplace), X); }
        
        public final Tensor[] hard_sigmoid(Tensor... X) { return fsp(new CoreHardSigmoid<>(sp_func, sp_math2_inplace), X); }
        public final Tensor[] hard_sigmoid(boolean inplace, Tensor... X) { return csp(inplace, new CoreHardSigmoid<>(sp_func, inplace), X); }

        public final Tensor[] tanh(Tensor... X) { return fsp(new CoreTanh<>(sp_func, sp_math2_inplace), X); }
        public final Tensor[] tanh(boolean inplace, Tensor... X) { return csp(inplace, new CoreTanh<>(sp_func, inplace), X); }

        public final Tensor[] softmax(Tensor... X) { return fsp(new CoreSoftmax<>(sp_func, sp_math2_inplace, -1), X); }
        public final Tensor[] softmax(int features, Tensor... X) {
            return fsp(new CoreSoftmax<>(sp_func, sp_math2_inplace, features), X);
        }
        public final Tensor[] softmax(boolean inplace, Tensor... X) { return csp(inplace, new CoreSoftmax<>(sp_func, inplace, -1), X); }
        public final Tensor[] softmax(boolean inplace, int features, Tensor... X) {
            return csp(inplace, new CoreSoftmax<>(sp_func, inplace, features), X);
        }
        
        public final Tensor[] log_softmax(Tensor... X) { return fsp(new CoreLogSoftmax<>(sp_func, sp_math2_inplace, -1), X); }
        public final Tensor[] log_softmax(int features, Tensor... X) {
            return fsp(new CoreLogSoftmax<>(sp_func, sp_math2_inplace, features), X);
        }
        public final Tensor[] log_softmax(boolean inplace, Tensor... X) { return csp(inplace, new CoreLogSoftmax<>(sp_func, inplace, -1), X); }
        public final Tensor[] log_softmax(boolean inplace, int features, Tensor... X) {
            return csp(inplace, new CoreLogSoftmax<>(sp_func, inplace, features), X);
        }

        public final Tensor[] halfSin(float Amp, Tensor... X) { return fsp(new CoreHalfSin<>(sp_func, sp_math2_inplace, Amp, 1.0f, 0.0f), X); }
        public final Tensor[] halfSin(float Amp, float alpha, float beta, Tensor... X) {
            return fsp(new CoreHalfSin<>(sp_func, sp_math2_inplace, Amp, alpha, beta), X);
        }
        public final Tensor[] halfSin(boolean inplace, float Amp, Tensor... X) { return csp(inplace, new CoreHalfSin<>(sp_func, inplace, Amp, 1.0f, 0.0f), X); }
        public final Tensor[] halfSin(boolean inplace, float Amp, float alpha, float beta, Tensor... X) {
            return csp(inplace, new CoreHalfSin<>(sp_func, inplace, Amp, alpha, beta), X);
        }
        
        public final Tensor[] tan(Tensor...X) { return fsp(new CoreTan<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] tan(float alpha, float beta, Tensor...X) { 
            return fsp(new CoreTan<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] tan(boolean inplace, Tensor...X) { return csp(inplace, new CoreTan<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] tan(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, new CoreTan<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] cot(Tensor...X) { return fsp(new CoreCot<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] cot(float alpha, float beta, Tensor...X) { 
            return fsp(new CoreCot<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] cot(boolean inplace, Tensor...X) { return csp(inplace, new CoreCot<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] cot(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, new CoreCot<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] arcsin(Tensor...X) { return fsp(new CoreArcsin<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X);  }
        public final Tensor[] arcsin(float alpha, float beta, Tensor...X) {
            return fsp(new CoreArcsin<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] arcsin(boolean inplace, Tensor...X) { return csp(inplace, new CoreArcsin<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] arcsin(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, new CoreArcsin<>(sp_func, inplace, alpha, beta), X);
        }
        
        public final Tensor[] arctan(Tensor...X) { return fsp(new CoreArctan<>(sp_func, sp_math2_inplace, 1.0f, 0.0f), X); }
        public final Tensor[] arctan(float alpha, float beta, Tensor...X) {
            return fsp(new CoreArctan<>(sp_func, sp_math2_inplace, alpha, beta), X); 
        }
        public final Tensor[] arctan(boolean inplace, Tensor...X) { return csp(inplace, new CoreArctan<>(sp_func, inplace, 1.0f, 0.0f), X); }
        public final Tensor[] arctan(boolean inplace, float alpha, float beta, Tensor...X) {
            return csp(inplace, new CoreArctan<>(sp_func, inplace, alpha, beta), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.tensor">
        public final Tensor[] flatten(Tensor... X) {  return fsp(new CoreFlatten<>(sp_func, sp_tensor_inplace), X); }
        public final Tensor[] flatten(boolean inplace, Tensor... X) {
            return csp(inplace, new CoreFlatten<>(sp_func, inplace), X);
        }
        
        public final Tensor[] view(Tensor[] X, int... out_dim) { 
            return fsp(new CoreView<>(sp_func, sp_tensor_inplace, out_dim), X); 
        }
        public final Tensor[] view(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreView<>(sp_func, inplace, out_dim), X);
        }
        
        public final Tensor[] view_flatten(Tensor... X) { 
            return fsp(new CoreView<>(sp_func, sp_tensor_inplace, X[0].dim(0), -1), X); 
        }
        public final Tensor[] view_flatten(boolean inplace, Tensor... X) {
            return csp(inplace, new CoreView<>(sp_func, inplace, X[0].dim(0), -1), X);
        }
        
        public final Tensor[] reshape(Tensor[] X, int... out_dim) { 
            return fsp(new CoreReshape<>(sp_func, sp_tensor_inplace, out_dim), X); 
        }
        public final Tensor[] reshape(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreReshape<>(sp_func, inplace, out_dim), X);
        }
        
        public final Tensor[] transpose(int dimIdx1, int dimIdx2, Tensor... X) {
            return fsp(new CoreTranspose<>(sp_func, sp_tensor_inplace, 
                    dimIdx1, dimIdx2), X);
        }
        public final Tensor[] transpose(boolean inplace, int dimIdx1, int dimIdx2, Tensor... X) {
            return csp(inplace, new CoreTranspose<>(sp_func, inplace, 
                    dimIdx1, dimIdx2), X);
        }
        
        public final Tensor[] rot180(Tensor... X) { 
            return fsp(new CoreRot180<>(sp_func, sp_tensor_inplace), X); 
        }
        public final Tensor[] rot180(boolean inplace, Tensor... X) {
            return csp(inplace, new CoreRot180<>(sp_func, inplace), X);
        }
        
        public final Tensor[] pad(Tensor[] X, int... p) { 
            return fsp(new CorePad<>(sp_func, sp_tensor_inplace, 
                    p, Vector.arrayCopy(p)), X); 
        }
        public final Tensor[] pad(Tensor[] X, int[] p0, int[] p1) { 
            return fsp(new CorePad<>(sp_func, sp_tensor_inplace, 
                    p0, p1), X); 
        }
        public final Tensor[] pad(boolean inplace, Tensor[] X, int... p) {
            return csp(inplace, new CorePad<>(sp_func, inplace, 
                    p, Vector.arrayCopy(p)), X);
        }
        public final Tensor[] pad(boolean inplace, Tensor[] X, int[] p0, int[] p1) {
            return csp(inplace, new CorePad<>(sp_func, inplace, 
                    p0, p1), X);
        }
        
        public final Tensor[] trim(Tensor[] X, int... t) { 
            return fsp(new CoreTrim<>(sp_func, sp_tensor_inplace, 
                    t, Vector.arrayCopy(t)), X); 
        }
        public final Tensor[] trim(Tensor[] X, int[] t0, int[] t1) { 
            return fsp(new CoreTrim<>(sp_func, sp_tensor_inplace, 
                    t0, t1), X); 
        }
        public final Tensor[] trim(boolean inplace, Tensor[] X, int... t) {
            return csp(inplace, new CoreTrim<>(sp_func, inplace, 
                    t, Vector.arrayCopy(t)), X);
        }
        public final Tensor[] trim(boolean inplace, Tensor[] X, int[] t0, int[] t1) {
            return csp(inplace, new CoreTrim<>(sp_func, inplace, 
                    t0, t1), X);
        }
        
        public final Tensor[] pad2D(Tensor[] X, int... p) { 
            return fsp(new CorePad<>(sp_func, sp_tensor_inplace, 
                    Vector.append(p, 0), Vector.append(p, 0)), X); 
        }
        public final Tensor[] pad2D(Tensor[] X, int[] p0, int[] p1) { 
            return fsp(new CorePad<>(sp_func, sp_tensor_inplace, 
                    Vector.append(p0, 0), Vector.append(p1, 0)), X); 
        }
        public final Tensor[] pad2D(boolean inplace, Tensor[] X, int... p) {
            return csp(inplace, new CorePad<>(sp_func, inplace, 
                    Vector.append(p, 0), Vector.append(p, 0)), X);
        }
        public final Tensor[] pad2D(boolean inplace, Tensor[] X, int[] p0, int[] p1) {
            return csp(inplace, new CorePad<>(sp_func, inplace, 
                    Vector.append(p0, 0), Vector.append(p1, 0)), X);
        }
        
        public final Tensor[] trim2D(Tensor[] X, int... t) { 
            return fsp(new CoreTrim<>(sp_func, sp_tensor_inplace, 
                    Vector.append(t, 0), Vector.append(t, 0)), X); 
        }
        public final Tensor[] trim2D(Tensor[] X, int[] t0, int[] t1) {
            return fsp(new CoreTrim<>(sp_func, sp_tensor_inplace, 
                    Vector.append(t0, 0), Vector.append(t1, 0)), X); 
        }
        public final Tensor[] trim2D(boolean inplace, Tensor[] X, int... t) {
            return csp(inplace, new CoreTrim<>(sp_func, inplace, 
                    Vector.append(t, 0), Vector.append(t, 0)), X);
        }
        public final Tensor[] trim2D(boolean inplace, Tensor[] X, int[] t0, int[] t1) {
            return csp(inplace, new CoreTrim<>(sp_func, inplace, 
                    Vector.append(t0, 0), Vector.append(t1, 0)), X);
        }
        
        public final Tensor[] expand(Tensor[] X, int... out_dim) {
            return fsp(new CoreExpand<>(sp_func, sp_tensor_inplace, 
                    Engine.from_center, out_dim), X);
        }
        public final Tensor[] expand(Tensor[] X, int[] start, int[] out_dim) {
            return fsp(new CoreExpand<>(sp_func, sp_tensor_inplace, 
                    start, out_dim), X);
        }
        public final Tensor[] expand(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreExpand<>(sp_func, inplace, 
                    Engine.from_center, out_dim), X);
        }
        public final Tensor[] expand(boolean inplace, Tensor[] X, int[] start, int[] out_dim) {
            return csp(inplace, new CoreExpand<>(sp_func, inplace, 
                    start, out_dim), X);
        }
        
        public final Tensor[] crop(Tensor[] X, int... out_dim) {
            return fsp(new CoreCrop<>(sp_func, sp_tensor_inplace, 
                    Engine.from_center, out_dim), X);
        }
        public final Tensor[] crop(Tensor[] X, int[] start, int[] out_dim) {
            return fsp(new CoreCrop<>(sp_func, sp_tensor_inplace, 
                    start, out_dim), X);
        }
        public final Tensor[] crop(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreCrop<>(sp_func, inplace, 
                    Engine.from_center, out_dim), X);
        }
        public final Tensor[] crop(boolean inplace, Tensor[] X, int[] start, int[] out_dim) {
            return csp(inplace, new CoreCrop<>(sp_func, inplace, 
                    start, out_dim), X);
        }
        
        public final Tensor[] expand2D(Tensor[] X, int... out_dim) {
            return fsp(new CoreExpand<>(sp_func, sp_tensor_inplace, 
                    Engine.from_center, Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] expand2D(Tensor[] X, int[] start, int[] out_dim) {
            return fsp(new CoreExpand<>(sp_func, sp_tensor_inplace, 
                    Vector.append(start, 0), Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] expand2D(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreExpand<>(sp_func, inplace, 
                    Engine.from_center, Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] expand2D(boolean inplace, Tensor[] X, int[] start, int[] out_dim) {
            return csp(inplace, new CoreExpand<>(sp_func, inplace, 
                    Vector.append(start, 0), Vector.append(out_dim, -1)), X);
        }
        
        public final Tensor[] crop2D(Tensor[] X, int... out_dim) {
            return fsp(new CoreCrop<>(sp_func, sp_tensor_inplace, 
                    Engine.from_center, Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] crop2D(Tensor[] X, int[] start, int[] out_dim) {
            return fsp(new CoreCrop<>(sp_func, sp_tensor_inplace, 
                    Vector.append(start, 0), Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] crop2D(boolean inplace, Tensor[] X, int... out_dim) {
            return csp(inplace, new CoreCrop<>(sp_func, inplace, 
                    Engine.from_center, Vector.append(out_dim, -1)), X);
        }
        public final Tensor[] crop2D(boolean inplace, Tensor[] X, int[] start, int[] out_dim) {
            return csp(inplace, new CoreCrop<>(sp_func, inplace, 
                    Vector.append(start, 0), Vector.append(out_dim, -1)), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.AvgPool2D">
        public final Tensor[] avgPool2D(int div, Tensor... X) { 
            return csp(new CoreAvgPool2D<>(sp_func, avgpool2D_igpad, 
                    div, div, div, div, 0, 0, 
                    -1, -1), X);
        }
        public final Tensor[] avgPool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp(new CoreAvgPool2D<>(sp_func, avgpool2D_igpad,
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1), X);
        }
        public final Tensor[] avgPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp(new CoreAvgPool2D<>(sp_func, avgpool2D_igpad,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width, 
                    -1, -1), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.MaxPool2D">
        public final Tensor[] maxPool2D(int div, Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, true, 
                    div, div, div, div, 0, 0, 
                    -1, -1), X);
        }
        public final Tensor[] maxPool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, true,
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1), X);
        }
        public final Tensor[] maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, true, 
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width,
                    -1, -1), X);
        }

        public final Tensor[] eval_maxPool2D(int div, Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, false, 
                    div, div, div, div, 0, 0, 
                    -1, -1), X);
        }
        public final Tensor[] eval_maxPool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, false,
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1), X);
        }
        public final Tensor[] eval_maxPool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp_maxPool(new CoreMaxPool2D<>(sp_func, false, 
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width,
                    -1, -1), X);
        }

        public final Tensor[] maxPool2D_with_Index(int div, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreMaxPool2D<>(sp_func, true, 
                    div, div, div, div, 0, 0,
                    -1, -1), X);
        }
        public final Tensor[] maxPool2D_with_Index(int kernel, int stride, int padding, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreMaxPool2D<>(sp_func, true, 
                    kernel, kernel, stride, stride, padding, padding, 
                    -1, -1), X);
        }
        public final Tensor[] maxPool2D_with_Index(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor... X) {
            return csp_maxPool_with_Idx(new CoreMaxPool2D<>(sp_func, true,
                    kernel_height, kernel_width,
                    stride_height, stride_width, 
                    padding_height, padding_width, 
                    -1, -1), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.AvgUnpool2d">
        public final Tensor[] avgUnpool2D(int mul, Tensor...X) { 
            return csp(new CoreAvgUnpool2D<>(sp_func, avgunpool2D_ignore_padding, 
                    mul, mul, mul, mul, 0, 0, 
                    -1, -1), X); 
        }
        public final Tensor[] avgUnpool2D(int kernel, int stride, int padding, Tensor... X) {
            return csp(new CoreAvgUnpool2D<>(sp_func, avgunpool2D_ignore_padding, 
                    kernel, kernel, stride, stride, padding, padding,
                    -1, -1), X);
        }
        public final Tensor[] avgUnpool2D(
                int kernel_height, int kernel_width,
                int stride_height, int stride_width,
                int padding_height, int padding_width, 
                Tensor...X) {
            return csp(new CoreAvgUnpool2D<>(sp_func, avgunpool2D_ignore_padding, 
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    padding_height, padding_width, 
                    -1, -1), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.AdpativeAvgPool2D">
        public final Tensor[] adaptive_avgPool2D(int out_size, Tensor... X) {
            return csp(new CoreAdaptiveAvgPool2D<>(sp_func, avgpool2D_igpad,
                    out_size, out_size), X);
        }
        public final Tensor[] adaptive_avgPool2D(int out_height, int out_width, Tensor... X) {
            return csp(new CoreAdaptiveAvgPool2D<>(sp_func, avgpool2D_igpad,
                    out_height, out_width), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.AdaptiveMaxPool2D">
        public final Tensor[] adaptive_maxPool2D(int out_size, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool2D<>(sp_func, 
                    true, out_size, out_size), X);
        }
        public final Tensor[] adaptive_maxPool2D(int out_height, int out_width, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool2D<>(sp_func,
                    true, out_height, out_width), X);
        }
        
        public final Tensor[] eval_adaptive_maxPool2D(int out_size, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool2D<>(sp_func,
                    false, out_size, out_size), X);
        }
        public final Tensor[] eval_adaptive_maxPool2D(int out_height, int out_width, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool2D<>(sp_func,
                    false, out_height, out_width), X);
        }

        public final Tensor[] adaptive_maxPool2D_with_Index(int out_size, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreAdaptiveMaxPool2D<>(sp_func, 
                    true, out_size, out_size), X);
        }
        public final Tensor[] adaptive_maxPool2D_with_Index(int out_height, int out_width, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreAdaptiveMaxPool2D<>(sp_func, 
                    true, out_height, out_width), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.Pool1D">
        public final Tensor[] avgPool1D(int div, Tensor... X) { 
            return csp(new CoreAvgPool1D<>(sp_func, avgpool1D_igpad, div, div, 0,  -1), X);
        }
        public final Tensor[] avgPool1D(int kernel_width, int stride_width, int padding_width, Tensor... X) {
            return csp(new CoreAvgPool1D<>(sp_func, avgpool1D_igpad, kernel_width, stride_width, padding_width, -1), X);
        }
        
        public final Tensor[] maxPool1D(int div, Tensor... X) {
            return csp_maxPool(new CoreMaxPool1D<>(sp_func, true, div, div, 0, -1), X);
        }
        public final Tensor[] maxPool1D(int kernel_width, int stride_width, int padding_width, Tensor... X) {
            return csp_maxPool(new CoreMaxPool1D<>(sp_func, true, kernel_width, stride_width, padding_width, -1), X);
        }

        public final Tensor[] eval_maxPool1D(int div, Tensor... X) {
            return csp_maxPool(new CoreMaxPool1D<>(sp_func, false, div, div, 0, -1), X);
        }
        public final Tensor[] eval_maxPool1D(int kernel_width, int stride_width, int padding_width, Tensor... X) {
            return csp_maxPool(new CoreMaxPool1D<>(sp_func, false,  kernel_width, stride_width, padding_width, -1), X);
        }

        public final Tensor[] maxPool1D_with_Index(int div, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreMaxPool1D<>(sp_func, true, div, div, 0,-1), X);
        }
        public final Tensor[] maxPool1D_with_Index(int kernel_width, int stride_width, int padding_width, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreMaxPool1D<>(sp_func, true, kernel_width, stride_width, padding_width, -1), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: simple.pool.AdpativeAvgPool1D">
        public final Tensor[] adaptive_avgPool1D(int out_width, Tensor... X) {
            return csp(new CoreAdaptiveAvgPool1D<>(sp_func, avgpool1D_igpad, out_width), X);
        }
        
        public final Tensor[] adaptive_maxPool1D(int out_width, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool1D<>(sp_func, true, out_width), X);
        }
        public final Tensor[] eval_adaptive_maxPool1D(int out_width, Tensor... X) {
            return csp_maxPool(new CoreAdaptiveMaxPool1D<>(sp_func, false, out_width), X);
        }
        public final Tensor[] adaptive_maxPool1D_with_Index(int out_width, Tensor... X) {
            return csp_maxPool_with_Idx(new CoreAdaptiveMaxPool1D<>(sp_func, true, out_width), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functioal: dual.blas">
        public final Tensor[] matMul(Tensor...X) { return cdu(new CoreMatMul<>(du_func), X); }
        public final Tensor[] matMulT1(Tensor...X) { return cdu(new CoreMatMulT1<>(du_func), X); }
        public final Tensor[] matMulT2(Tensor...X) { return cdu(new CoreMatMulT2<>(du_func), X); }
        
        public final Tensor[] batchMatMul(Tensor...X) { return cdu(new CoreBatchMatMul<>(du_func, dl_likeX1), X); }
        public final Tensor[] batchMatMul(boolean likeX1, Tensor...X) { return cdu(new CoreBatchMatMul<>(du_func, likeX1), X); }
        
        public final Tensor[] batchMatMulT1(Tensor...X) { return cdu(new CoreBatchMatMulT1<>(du_func, dl_likeX1), X); }
        public final Tensor[] batchMatMulT1(boolean likeX1, Tensor...X) { return cdu(new CoreBatchMatMulT1<>(du_func, likeX1), X); }
       
        public final Tensor[] batchMatMulT2(Tensor... X) { return cdu(new CoreBatchMatMulT2<>(du_func, dl_likeX1), X); }
        public final Tensor[] batchMatMulT2(boolean likeX1, Tensor... X) { return cdu(new CoreBatchMatMulT2<>(du_func, likeX1), X); }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functioal: dual.math">
        public final Tensor[] add(Tensor... X) { return cdu(new CoreLinear2<>(du_func, dl_likeX1, 1.0f,  1.0f, 0.0f), X); }
        public final Tensor[] sub(Tensor... X) { return cdu(new CoreLinear2<>(du_func, dl_likeX1, 1.0f, -1.0f, 0.0f), X); }
        public final Tensor[] add(float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2<>(du_func, dl_likeX1, alpha,  beta, 0.0f), X); }
        public final Tensor[] sub(float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2<>(du_func, dl_likeX1, alpha, -beta, 0.0f), X); }
        public final Tensor[] linear2(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2<>(du_func, dl_likeX1, alpha, beta, gamma), X);
        }
        
        public Tensor[] add(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] sub(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2<>(du_func, likeX1,  1.0f, -1.0f, 0.0f), X); }
        public Tensor[] add(boolean likeX1, float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2<>(du_func, likeX1, alpha,  beta, 0.0f), X); }
        public Tensor[] sub(boolean likeX1, float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2<>(du_func, likeX1, alpha, -beta, 0.0f), X); }
        public Tensor[] linear2(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        
        public final Tensor[] add_row(Tensor... X) { return cdu(new CoreLinear2Row<>(du_func, 1.0f,  1.0f, 0.0f), X); }
        public final Tensor[] sub_row(Tensor... X) { return cdu(new CoreLinear2Row<>(du_func, 1.0f, -1.0f, 0.0f), X); }
        public final Tensor[] add_row(float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2Row<>(du_func, alpha,  beta, 0.0f), X); }
        public final Tensor[] sub_row(float alpha, float beta, Tensor... X) { return cdu(new CoreLinear2Row<>(du_func, alpha, -beta, 0.0f), X); }
        public final Tensor[] linear2_row(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2Row<>(du_func, alpha, beta, gamma), X);
        }
        
        public Tensor[] mul(Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] mul(float alpha, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd(Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqsub(Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd(float alpha, float beta, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] quadratic2(float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(new CoreQuadratic2<>(du_func, dl_likeX1, k11, k12, k22, k1, k2, C), X);
        }
        
        public Tensor[] mul(boolean likeX1, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, likeX1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] mul(boolean likeX1, float alpha, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, likeX1, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd(boolean likeX1, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, likeX1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqsub(boolean likeX1, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, likeX1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd(boolean likeX1, float alpha, float beta, Tensor... X) { return cdu(new CoreQuadratic2<>(du_func, likeX1, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] quadratic2(boolean likeX1, float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(new CoreQuadratic2<>(du_func, likeX1, k11, k12, k22, k1, k2, C), X);
        }
        
        public Tensor[] mul_row(Tensor... X) { return cdu(new CoreQuadratic2Row<>(du_func, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] mul_row(float alpha, Tensor... X) { return cdu(new CoreQuadratic2Row<>(du_func, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd_row(Tensor... X) { return cdu(new CoreQuadratic2Row<>(du_func, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqsub_row(Tensor... X) { return cdu(new CoreQuadratic2Row<>(du_func, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd_row(float alpha, float beta, Tensor... X) { return cdu(new CoreQuadratic2Row<>(du_func, alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] quadratic2_row(float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(new CoreQuadratic2Row<>(du_func, k11, k12, k22, k1, k2, C), X);
        }
        
        public Tensor[] mul_center(Tensor... X) { return cdu(new CoreQuadratic2Center<>(du_func, -1, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] mul_center(float alpha, Tensor... X) { return cdu(new CoreQuadratic2Center<>(du_func, -1, 0.0f, alpha, 0.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd_center(Tensor... X) { return cdu(new CoreQuadratic2Center<>(du_func, -1, 1.0f, 0.0f,  1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqsub_center(Tensor... X) { return cdu(new CoreQuadratic2Center<>(du_func, -1, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] sqadd_center(float alpha, float beta, Tensor... X) { return cdu(new CoreQuadratic2Center<>(du_func, -1,  alpha, 0.0f, beta, 0.0f, 0.0f, 0.0f), X); }
        public Tensor[] quadratic2_center(float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) { 
            return cdu(new CoreQuadratic2Center<>(du_func, -1, k11, k12, k22, k1, k2, C), X);
        }
        public Tensor[] quadratic2_center(int dim2, float k11, float k12, float k22, float k1, float k2, float C, Tensor... X) {
            return cdu(new CoreQuadratic2Center<>(du_func, dim2, k11, k12, k22, k1, k2, C), X);
        }
        
        public Tensor[] div(Tensor... X) { return cdu(new CoreDiv<>(du_func, dl_likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f), X); }
        public Tensor[] div(float alpha1, float beta1, float alpha2, float beta2, float gamma, Tensor... X) {
            return cdu(new CoreDiv<>(du_func, dl_likeX1, alpha1, beta1, alpha2, beta2, gamma), X);
        }
        public Tensor[] div(boolean likeX1, Tensor... X) { return cdu(new CoreDiv<>(du_func, likeX1, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f), X); }
        public Tensor[] div(boolean likeX1, float alpha1, float beta1, float alpha2, float beta2, float gamma, Tensor... X){
            return cdu(new CoreDiv<>(du_func, likeX1, alpha1, beta1, alpha2, beta2, gamma), X);
        }
        
        public Tensor[] add_relu(Tensor... X) { return cdu(new CoreLinear2_Relu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_relu(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Relu<>(du_func, dl_likeX1, alpha, beta, gamma), X); 
        }
        public Tensor[] add_relu(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Relu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_relu(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Relu<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        
        public Tensor[] add_leakyRelu(Tensor... X) { return cdu(new CoreLinear2_LeakyRelu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f, leakyRelu_neg_slope), X); }
        public Tensor[] add_leakyRelu(float negative_slope, Tensor... X) { return cdu(new CoreLinear2_LeakyRelu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f, negative_slope), X); }
        public Tensor[] linear2_leakyRelu(float alpha, float beta, float gamma, Tensor... X) { 
            return cdu(new CoreLinear2_LeakyRelu<>(du_func, dl_likeX1, alpha, beta, gamma, leakyRelu_neg_slope), X);
        }
        public Tensor[] linear2_leakyRelu(float alpha, float beta, float gamma, float negative_slope, Tensor... X) {
            return cdu(new CoreLinear2_LeakyRelu<>(du_func, dl_likeX1, alpha, beta, gamma, negative_slope), X);
        }
        public Tensor[] add_leakyRelu(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_LeakyRelu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f, leakyRelu_neg_slope), X); }
        public Tensor[] add_leakyRelu(boolean likeX1, float negative_slope, Tensor... X) { return cdu(new CoreLinear2_LeakyRelu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f, negative_slope), X); }
        public Tensor[] linear2_leakyRelu(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_LeakyRelu<>(du_func, likeX1, alpha, beta, gamma, leakyRelu_neg_slope), X);
        }
        public Tensor[] linear2_leakyRelu(boolean likeX1, float alpha, float beta, float gamma, float negative_slope, Tensor... X) {
            return cdu(new CoreLinear2_LeakyRelu<>(du_func, likeX1, alpha, beta, gamma, negative_slope), X);
        }
        
        public Tensor[] add_elu(Tensor... X) { return cdu(new CoreLinear2_Elu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f, elu_alpha, elu_neg_slope), X); }
        public Tensor[] add_elu(float theta, float negative_slope, Tensor... X) { return cdu(new CoreLinear2_Elu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f, theta, negative_slope), X); }
        public Tensor[] linear2_elu(float alpha, float beta, float gamma, Tensor... X) { 
            return cdu(new CoreLinear2_Elu<>(du_func, dl_likeX1, alpha, beta, gamma, elu_alpha, elu_neg_slope), X);
        }
        public Tensor[] linear2_elu(float alpha, float beta, float gamma, float theta, float negative_slope, Tensor... X) {
            return cdu(new CoreLinear2_Elu<>(du_func, dl_likeX1, alpha, beta, gamma, theta, negative_slope), X);
        }
        public Tensor[] add_elu(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Elu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f, elu_alpha, elu_neg_slope), X); }
        public Tensor[] add_elu(boolean likeX1, float theta, float negative_slope, Tensor... X) { return cdu(new CoreLinear2_Elu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f, theta, negative_slope), X); }
        public Tensor[] linear2_elu(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Elu<>(du_func, likeX1, alpha, beta, gamma, elu_alpha, elu_neg_slope), X);
        }
        public Tensor[] linear2_elu(boolean likeX1, float alpha, float beta, float gamma, float theta, float negative_slope, Tensor... X) {
            return cdu(new CoreLinear2_Elu<>(du_func, likeX1, alpha, beta, gamma, theta, negative_slope), X);
        }
        
        public Tensor[] add_softplus(Tensor... X) { return cdu(new CoreLinear2_Softplus<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_softplus(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Softplus<>(du_func, dl_likeX1, alpha, beta, gamma), X);
        }
        public Tensor[] add_softplus(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Softplus<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_softplus(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Softplus<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        
        public Tensor[] add_gelu(Tensor... X) { return cdu(new CoreLinear2_Gelu<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_gelu(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Gelu<>(du_func, dl_likeX1, alpha, beta, gamma), X);
        }
        public Tensor[] add_gelu(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Gelu<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_gelu(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Gelu<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        
        public Tensor[] add_sigmoid(Tensor... X) { return cdu(new CoreLinear2_Sigmoid<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_sigmoid(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Sigmoid<>(du_func, dl_likeX1, alpha, beta, gamma), X);
        }
        public Tensor[] add_sigmoid(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Sigmoid<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_sigmoid(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Sigmoid<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        
        public Tensor[] add_tanh(Tensor... X) { return cdu(new CoreLinear2_Tanh<>(du_func, dl_likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_tanh(float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Tanh<>(du_func, dl_likeX1, alpha, beta, gamma), X);
        }
        public Tensor[] add_tanh(boolean likeX1, Tensor... X) { return cdu(new CoreLinear2_Tanh<>(du_func, likeX1, 1.0f, 1.0f, 0.0f), X); }
        public Tensor[] linear2_tanh(boolean likeX1, float alpha, float beta, float gamma, Tensor... X) {
            return cdu(new CoreLinear2_Tanh<>(du_func, likeX1, alpha, beta, gamma), X);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: reducer.math">
        public Tensor[] sum(Tensor... X) { 
            return cre(new CoreLinearSummary<>(re_func, 1.0f, 0.0f), X); 
        }
        public Tensor[] sum(float alpha, Tensor... X) {
            return cre(new CoreLinearSummary<>(re_func, alpha, 0.0f), X); 
        }
        public Tensor[] linearSum(float alpha, float beta, Tensor... X) {
            return cre(new CoreLinearSummary<>(re_func, alpha, beta), X); 
        }

        public Tensor[] mean(Tensor... X) { 
            return cre(new CoreLinearMean<>(re_func, 1.0f, 0.0f), X); 
        }
        public Tensor[] mean(float alpha, Tensor... X) { 
            return cre(new CoreLinearMean<>(re_func, alpha, 0.0f), X); 
        }
        public Tensor[] linearMean(float alpha, float beta, Tensor... X) { 
            return cre(new CoreLinearMean<>(re_func, alpha, beta), X); 
        }
        
        public Tensor[] squareSum(Tensor... X) {
            return cre(new CoreQuadraticSummary<>(re_func, 1.0f, 0.0f, 0.0f), X);
        }
        public Tensor[] squareSum(float alpha, Tensor... X) {
            return cre(new CoreQuadraticSummary<>(re_func, alpha, 0.0f, 0.0f), X);
        }
        public Tensor[] quadraticSum(float alpha, float beta, float gamma, Tensor... X) {
            return cre(new CoreQuadraticSummary<>(re_func, alpha, beta, gamma), X);
        }
        
        public Tensor[] squareMean(Tensor... X) {
            return cre(new CoreQuadraticMean<>(re_func, 1.0f, 0.0f, 0.0f), X);
        }
        public Tensor[] squareMean(float alpha, Tensor... X) {
            return cre(new CoreQuadraticMean<>(re_func, alpha, 0.0f, 0.0f), X);
        }
        public Tensor[] quadraticMean(float alpha, float beta, float gamma, Tensor... X) {
            return cre(new CoreQuadraticMean<>(re_func, alpha, beta, gamma), X);
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="functional: reducer.tensor">
        public Tensor[] concat(Tensor... X) { return cre(new CoreConcat<>(re_func, -1), X); }
        public Tensor[] concat(int dimIdx, Tensor... X) { return cre(new CoreConcat(re_func, dimIdx), X); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="functional: furcartion.tensor">
        public Tensor[] split(Tensor[] X, int...section) { 
            return cfc(new CoreSplit(fc_func, -1, section), X); 
        }
        public Tensor[] split(int dimIdx, Tensor[] X, int... section) {
            return cfc(new CoreSplit(fc_func, dimIdx, section), X);
        }
        
        public Tensor[] chunk(int n, Tensor... X) {
            return cfc(new CoreChunk(fc_func, -1, n), X);
        }
        public Tensor[] chunk(int dimIdx, int n, Tensor... X) {
            return cfc(new CoreChunk(fc_func, dimIdx, n), X);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Loss">
    public static class Loss {
        protected Loss() {}
        public static final Loss loss = new Loss();
        
        //<editor-fold defaultstate="collapsed" desc="Norm Loss">
        public L1 L1() { return new L1(); }
        public L2 L2() { return new L2(); }
        public SmoothL1 SmoothL1() { return new SmoothL1(); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="Binary Cross Entropy">
        public BinaryCrossEntropy binaryCrossEntropy() { 
            return new BinaryCrossEntropy(1.0f, 1.0f);
        }
        public BinaryCrossEntropy binaryCrossEntropy(float alpha, float beta) { 
            return new BinaryCrossEntropy(alpha, beta);
        }
        
        public SigmoidBinaryCrossEntropy sigmoid_binaryCrossEntropy() {
            return new SigmoidBinaryCrossEntropy(1.0f, 1.0f);
        }
        public SigmoidBinaryCrossEntropy sigmoid_binaryCrossEntropy(float alpha, float beta) {
            return new SigmoidBinaryCrossEntropy(alpha, beta);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="CrossEntropy">
        public CrossEntropy crossEntropy() { return new CrossEntropy(); }
       
        public SoftmaxCrossEntropy softmax_crossEntropy() { return new SoftmaxCrossEntropy(-1); }
        public SoftmaxCrossEntropy softmax_crossEntropy(int features) {
            return new SoftmaxCrossEntropy(features);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="class: LossSummaryCreator">
        public static final class WeightedSummaryCreator {
            private final List<Float> weights = new ArrayList<>(4);
            private final List<LossFunction> loss_funcs = new ArrayList<>(4);
            
            private WeightedSummaryCreator() {}
            
            public List<Float> weights() { return weights; }
            public List<LossFunction> loss_funcs() { return loss_funcs; }
            
            public WeightedSummaryCreator append(float weight, LossFunction loss) {
                if(loss == null) throw new NullPointerException("loss is null");
                weights.add(weight);
                loss_funcs.add(loss);
                return this;
            }
            
            public WeightedSummaryCreator clear() {
                weights.clear();
                loss_funcs.clear();
                return this;
            }
            
            public WeightedSummary create() 
            {
                float[] w = new float[weights.size()]; {
                    int index = 0; 
                    for(float weight : weights) w[index++] = weight;
                }
                
                LossFunction[] lf = new LossFunction[loss_funcs.size()]; {
                    int index = 0;
                    for(LossFunction loss : loss_funcs) lf[index++] = loss;
                }
                
                return new WeightedSummary(w, lf);
            }
        }
        //</editor-fold>
        public WeightedSummaryCreator weighted_sum() { return new WeightedSummaryCreator(); }
        public WeightedSummary weighted_sum(float[] weight, LossFunction[] loss) {
            return new WeightedSummary(weight, loss);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Opim">
    public static class Optim 
    {
        protected Optim() {}
        public static final Optim optim = new Optim();

        //<editor-fold defaultstate="collpased" desc="create: Optimizer">
        public SGD SGD(Collection<Parameter> params, float lr) { return new SGD(params, lr);  }
        public SGD SGD(Map<String, Parameter> params, float lr) { return new SGD(params, lr); }
        
        public static float Momentum_beta = 0.9f;
        
        public Momentum Momentum(Collection<Parameter> params, float lr) { 
            return new Momentum(params, lr, Momentum_beta); 
        }
        public Momentum Momentum(Collection<Parameter> params, float lr, float beta) {
            return new Momentum(params, lr, beta);
        }
        
        public Momentum Momentum(Map<String, Parameter> param_map, float lr) { 
            return new Momentum(param_map, lr, Momentum_beta); 
        }
        public Momentum Momentum(Map<String, Parameter> param_map, float lr, float beta) {
            return new Momentum(param_map, lr, beta);
        }
        
        
        public static float SGDMN_momentum = 0.9f;
        public static float SGDMN_dampen = 0.0f;
        public static float nestrov = 0.0f;
        
        public SGDMN SGDMN(Collection<Parameter> params, float lr) {
            return new SGDMN(params, lr, SGDMN_momentum, SGDMN_dampen, nestrov);
        }
        public SGDMN SGDMN(Collection<Parameter> params, float lr, float momentum, float dampen, float nestrov) {
            return new SGDMN(params, lr, momentum, dampen, nestrov);
        }
        
        public SGDMN SGDMN(Map<String, Parameter> param_map, float lr) {
            return new SGDMN(param_map, lr, SGDMN_momentum, SGDMN_dampen, nestrov);
        }
        public SGDMN SGDMN(Map<String, Parameter> param_map, float lr, float momentum, float dampen, float nestrov) {
            return new SGDMN(param_map, lr, momentum, dampen, nestrov);
        }
        

        public static float RMSprop_lr = 1e-2f;
        public static float RMSprop_beta = 0.99f;
        public static float RMSprop_eps = 1e-8f;
        
        public RMSprop RMSprop(Collection<Parameter> params) {
            return new RMSprop(params, RMSprop_lr, RMSprop_beta, RMSprop_eps);
        }
        public RMSprop RMSprop(Collection<Parameter> params, float lr) { 
            return new RMSprop(params, lr, RMSprop_beta, RMSprop_eps);
        }
        public RMSprop RMSprop(Collection<Parameter> params, float lr, float beta, float eps) {
            return new RMSprop(params, lr, beta, eps);
        }
        
        public RMSprop RMSprop(Map<String, Parameter> param_map) {
            return new RMSprop(param_map, RMSprop_lr, RMSprop_beta, RMSprop_eps);
        }
        public RMSprop RMSprop(Map<String, Parameter> param_map, float lr) { 
            return new RMSprop(param_map, lr, RMSprop_beta, RMSprop_eps);
        }
        public RMSprop RMSprop(Map<String, Parameter> param_map, float lr, float beta, float eps) {
            return new RMSprop(param_map, lr, beta, eps);
        }
        
        
        public static float Adam_lr = 1e-3f;
        public static float Adam_beta1 = 0.9f;
        public static float Adam_beta2 = 0.999f;
        public static float Adam_eps = 1e-8f;
        
        public Adam Adam(Collection<Parameter> params) { 
            return new Adam(params, Adam_lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Collection<Parameter> params, float lr) { 
            return new Adam(params, lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
            return new Adam(params, lr, beta1, beta2, eps);
        }       
        
        public Adam Adam(Map<String, Parameter> param_map) { 
            return new Adam(param_map, Adam_lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Map<String, Parameter> param_map, float lr) { 
            return new Adam(param_map, lr, Adam_beta1, Adam_beta2, Adam_eps); 
        }
        public Adam Adam(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
            return new Adam(param_map, lr, beta1, beta2, eps);
        }       
        
        
        public static float Adamax_lr = 2e-3f;
        public static float Adamax_beta1 = 0.9f;
        public static float Adamax_beta2 = 0.999f;
        public static float Adamax_eps = 1e-8f;
        
        public Adamax Adamax(Collection<Parameter> params) { 
            return new Adamax(params, Adamax_lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Collection<Parameter> params, float lr) { 
            return new Adamax(params, lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
            return new Adamax(params, lr, beta1, beta2, eps);
        }
        
        public Adamax Adamax(Map<String, Parameter> param_map) { 
            return new Adamax(param_map, Adamax_lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Map<String, Parameter> param_map, float lr) { 
            return new Adamax(param_map, lr, Adamax_beta1, Adamax_beta2, Adamax_eps); 
        }
        public Adamax Adamax(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
            return new Adamax(param_map, lr, beta1, beta2, eps);
        }
       
        
        public static float AdamW_lr = 1e-3f;
        public static float AdamW_beta1 = 0.9f;
        public static float AdamW_beta2 = 0.999f;
        public static float AdamW_eps = 1e-8f;
        public static float AdamW_L1 = 0;
        public static float AdamW_L2 = 1e-2f;
        
        public AdamW AdamW(Collection<Parameter> params) {
            return new AdamW(params, AdamW_lr, AdamW_beta1, AdamW_beta2, AdamW_eps, AdamW_L1, AdamW_L2); 
        }
        public AdamW AdamW(Collection<Parameter> params, float lr, float L2) {
            return new AdamW(params, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, AdamW_L1, L2); 
        }
        public AdamW AdamW(Collection<Parameter> params, float lr, 
                float beta1, float beta2, float eps,
                float L1, float L2) {
            return new AdamW(params, lr, beta1, beta2, eps, L1, L2);
        }
        
        public AdamW AdamW(Map<String, Parameter> param_map) {
            return new AdamW(param_map, AdamW_lr, AdamW_beta1, AdamW_beta2, AdamW_eps, AdamW_L1, AdamW_L2); 
        }
        public AdamW AdamW(Map<String, Parameter> param_map, float lr, float L2) {
            return new AdamW(param_map, lr, AdamW_beta1, AdamW_beta2, AdamW_eps, AdamW_L1, L2); 
        }
        public AdamW AdamW(Map<String, Parameter> param_map, float lr, 
                float beta1, float beta2, float eps,
                float L1, float L2) {
            return new AdamW(param_map, lr, beta1, beta2, eps, L1, L2);
        }
        
        
        public static float RAdam_lr = 1e-3f;
        public static float RAdam_beta1 = 0.9f;
        public static float RAdam_beta2 = 0.999f;
        public static float RAdam_eps = 1e-8f;
        
        public RAdam RAdam(Collection<Parameter> params) { 
            return new RAdam(params, RAdam_lr, RAdam_beta1, RAdam_beta2, RAdam_eps); 
        }
        public RAdam RAdam(Collection<Parameter> params, float lr) { 
            return new RAdam(params, lr, RAdam_beta1, RAdam_beta2, RAdam_eps); 
        }
        public RAdam RAdam(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
            return new RAdam(params, lr, beta1, beta2, eps);
        }       
        
        public RAdam RAdam(Map<String, Parameter> param_map) { 
            return new RAdam(param_map, RAdam_lr, RAdam_beta1, RAdam_beta2, RAdam_eps); 
        }
        public RAdam RAdam(Map<String, Parameter> param_map, float lr) { 
            return new RAdam(param_map, lr, RAdam_beta1, RAdam_beta2, RAdam_eps); 
        }
        public RAdam RAdam(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
            return new RAdam(param_map, lr, beta1, beta2, eps);
        }       
        
        public static float Adamod_lr = 1e-3f;
        public static float Adamod_beta1 = 0.9f;
        public static float Adamod_beta2 = 0.999f;
        public static float Adamod_beta3 = 0.999f;
        public static float Adamod_eps = 1e-8f;
        
        public Adamod Adamod(Collection<Parameter> params) { 
            return new Adamod(params, Adamod_lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Collection<Parameter> params, float lr) { 
            return new Adamod(params, lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Collection<Parameter> params, float lr, float beta1, float beta2, float eps, float beta3) {
            return new Adamod(params, lr, beta1, beta2, eps, beta3);
        }
        
        public Adamod Adamod(Map<String, Parameter> param_map) { 
            return new Adamod(param_map, Adamod_lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Map<String, Parameter> param_map, float lr) { 
            return new Adamod(param_map, lr, Adamod_beta1, Adamod_beta2, Adamod_eps, Adamod_beta3); 
        }
        public Adamod Adamod(Map<String, Parameter> param_map, float lr,  float beta1, float beta2, float eps, float beta3) {
            return new Adamod(param_map, lr, beta1, beta2, eps, beta3);
        }
        //</editor-fold>

        //<editor-fold defaultstate="collpased" desc="creat: LrSchedular">
        public CosAnnealingLr cosAnnealingLr(float Tmax) { return new CosAnnealingLr(Tmax, 0f); }
        public CosAnnealingLr cosAnnealingLr(float Tmax, float minLr) { return new CosAnnealingLr(Tmax, minLr); }

        public ExponentialLr exponentialLr(float gamma, float minLr) { return new ExponentialLr(gamma, minLr); }

        public LambdaLr lambdaLr(Function<Float, Float> updater) { return new LambdaLr(updater); }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Datas">
    public static class Datas 
    {
        protected Datas() {}
        public static final Datas data = new Datas();
        
        //<editor-fold defaultstate="collapsed" desc="create: AutoLoadContainer">
        public static int autoLoad_capcity = 2048;
        public static int autoLoad_thread_num = 4;
        public static double autoLoad_update_threshold = 0.5f;
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_clazz, Class<V> label_clazz) {
            return new AutoLoadContainer<>(input_clazz, label_clazz, autoLoad_capcity)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_clazz, Class<V> label_clazz,
                Loader<K, V> loader) {
            return new AutoLoadContainer<>(input_clazz, label_clazz, autoLoad_capcity)
                    .loader(loader)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_clazz, Class<V> label_clazz, 
                Loader<K, V> loader, Triger triger) {
            return new AutoLoadContainer<>(input_clazz, label_clazz, autoLoad_capcity)
                    .loader(loader)
                    .triger(triger)
                    .thread_num(autoLoad_thread_num);
        }
        
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_clazz, Class<V> label_clazz, int capacity)  {
            return new AutoLoadContainer<>(input_clazz, label_clazz, capacity)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                Loader<K, V> loader) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(AutoLoadContainer.update(autoLoad_update_threshold))
                    .thread_num(autoLoad_thread_num);
        }
        public <K, V> AutoLoadContainer<K, V> autoLoad(Class<K> input_cls, Class<V> label_cls, int capacity, 
                Loader<K, V> loader, Triger triger) {
            return new AutoLoadContainer<>(input_cls, label_cls, capacity)
                    .loader(loader)
                    .triger(triger)
                    .thread_num(autoLoad_thread_num);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: ListContainer">
        public static int list_initCapacity = 2048;
        
        public <K, V> ListContainer<K, V> list(Class<K> input_cls, Class<V> label_cls) {
            return new ListContainer<>(input_cls, label_cls, list_initCapacity);
        }
        public <K, V> ListContainer<K, V> list(Class<K> input_cls, Class<V> label_cls, int initCapacity) {
            return new ListContainer<>(input_cls, label_cls, initCapacity);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: DataSet">
        public<K, V> DataSet<K, V> dataset(DataContainer<K, V> conta) { return new DataSet<>(conta); }
        public<K, V> DataSet<K, V> dataset(DataContainer<K, V> conta, 
                Transform<K[]> input_transform, 
                Transform<V[]> label_transform) {
            return new DataSet<>(conta)
                    .input_transform(input_transform)
                    .label_transform(label_transform);
        }
        
        public DataSet<byte[], Integer> minist(boolean train) {  return (train ? Minist.train() : Minist.test()); }
        public DataSet<byte[], Integer> minist(boolean train, String dir) { return (train ? Minist.train(dir) : Minist.test(dir)); }
        public DataSet<byte[], Integer> minist(boolean train, Transform<byte[][]> input_transform) { 
            return (train ? Minist.train(input_transform) : Minist.test(input_transform));
        }
        public DataSet<byte[], Integer> minist(boolean train, String dir, Transform<byte[][]> input_transform) {
            return (train ? Minist.train(dir, input_transform) : Minist.test(dir, input_transform));
        }

        public DataSet<byte[], Integer> cifar10(boolean train) { return (train ? Cifar10.train() : Cifar10.test()); }
        public DataSet<byte[], Integer> cifar10(boolean train, String dir) { return (train ? Cifar10.train(dir) : Cifar10.test(dir)); }
        public DataSet<byte[], Integer> cifar10(boolean train, Transform<byte[][]> input_transform) { 
            return (train ? Cifar10.train(input_transform) : Cifar10.test(input_transform));
        }
        public DataSet<byte[], Integer> cifar10(boolean train, String dir, Transform<byte[][]> input_transform) {
            return (train ? Cifar10.train(dir, input_transform) : Cifar10.test(dir, input_transform));
        }
        
        public DataSet<byte[], Integer> cifar100(boolean train) { return (train ? Cifar100.train() : Cifar100.test()); }
        public DataSet<byte[], Integer> cifar100(boolean train, String dir) { return (train ? Cifar100.train(dir) : Cifar100.test(dir)); }
        public DataSet<byte[], Integer> cifar100(boolean train, Transform<byte[][]> input_transform) { 
            return (train ? Cifar100.train(input_transform) : Cifar100.test(input_transform));
        }
        public DataSet<byte[], Integer> cifar100(boolean train, String dir, Transform<byte[][]> input_transform) {
            return (train ? Cifar100.train(dir, input_transform) : Cifar100.test(dir, input_transform));
        }
        
        public FileFolder file_folder(String root_dir) { return new FileFolder(list(File.class, Integer.class), new File(root_dir), null); }
        public FileFolder file_folder(String root_dir, Map<String, Integer> labels) {
            return new FileFolder(list(File.class, Integer.class), new File(root_dir), labels);
        }
        public FileFolder file_folder(File root_dir) { return new FileFolder(list(File.class, Integer.class), root_dir, null); }
        public FileFolder file_folder(File root_dir, Map<String, Integer> labels) {
            return new FileFolder(list(File.class, Integer.class), root_dir, labels);
        }
        
        public ImageFolder image_folder(String root_dir) { return new ImageFolder(file_folder(root_dir), null, -1); }
        public ImageFolder image_folder(String root_dir, Function<File, byte[]> pixel_transform) {
            return new ImageFolder(file_folder(root_dir), pixel_transform, -1);
        }
        public ImageFolder image_folder(String root_dir, Function<File, byte[]> pixel_transform, int num_threads) {
            return new ImageFolder(file_folder(root_dir), pixel_transform, num_threads);
        }
        public ImageFolder image_folder(String root_dir, Map<String, Integer> labels,
                Function<File, byte[]> pixel_transform, int num_threads) {
            return new ImageFolder(file_folder(root_dir, labels), pixel_transform, num_threads);
        }
        
        public ImageFolder image_folder(File root_dir) { return new ImageFolder(file_folder(root_dir), null, -1); }
        public ImageFolder image_folder(File root_dir, Function<File, byte[]> pixel_transform) {
            return new ImageFolder(file_folder(root_dir), pixel_transform, -1);
        }
        public ImageFolder image_folder(File root_dir, Function<File, byte[]> pixel_transform, int num_threads) {
            return new ImageFolder(file_folder(root_dir), pixel_transform, num_threads);
        }
        public ImageFolder image_folder(File root_dir, Map<String, Integer> labels,
                Function<File, byte[]> pixel_transform, int num_threads) {
            return new ImageFolder(file_folder(root_dir, labels), pixel_transform, num_threads);
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: Transform">
        //<editor-fold defaultstate="collapsed" desc="class: PixelToTensor: byte[][] -> Tensor"> 
        public static class PixelToTensor implements Transform<byte[][]> 
        {
            private final int[] dim;
            
            private PixelToTensor(int...dim) { this.dim = dim; }
            
            public final int[] dim() { return dim; }

            @Override
            public final Tensor transform(Engine eg, byte[][] value) {
                return eg.pix_to_tensor(value, dim);
            }
        }
        //</editor-fold>
        public PixelToTensor pixel_to_tensor(int... dim) { return new PixelToTensor(dim);  }

        //<editor-fold defaultstate="collapsed" desc="class: FloatToTensor: float[][] -> Tensor"> 
        public static class FloatToTensor implements Transform<float[][]> 
        {
            private final int[] dim;
            
            public FloatToTensor(int... dim) { this.dim = dim; }
            
            public final int[] dim() { return dim; }

            @Override
            public final Tensor transform(Engine eg, float[][] value) {
                return eg.tensor(value, dim);
            }
        }
        //</editor-fold>
        public FloatToTensor float_to_tensor(int... dim) { return new FloatToTensor(dim); }
        
        //<editor-fold defaultstate="collapsed" desc="class: IntToTensor: int[][] -> Tensor"> 
        public static class Int32ToTensor implements Transform<int[][]>
        {
            protected final int[] dim;
            protected final float alpha;
            protected final float beta;
            
            public Int32ToTensor(float alpha, float beta, int... dim) {
                this.alpha = alpha;
                this.beta = beta;
                this.dim = dim; 
            }

            public final float alpha() { return alpha; }
            public final float beta() { return beta; }
            public final int[] dim() { return dim; }
                    
            @Override
            public final Tensor transform(Engine eg, int[][] value) {
                return eg.tensor(alpha, value, beta, dim);
            }
        }
        //</editor-fold>
        public Int32ToTensor int32_to_tensor(float alpha, float beta, int[] dim) { return new Int32ToTensor(alpha, beta, dim); }
        public Int32ToTensor int32_to_tensor(int... dim) { return new Int32ToTensor(1.0f, 0.0f, dim); }
        
        //<editor-fold defaultstate="collapsed" desc="class: IntToTensor: byte[][] -> Tensor"> 
        public static class Int8ToTensor implements Transform<byte[][]>
        {
            protected final int[] dim;
            protected final float alpha;
            protected final float beta;
            
            public Int8ToTensor(float alpha, float beta, int... dim) {
                this.alpha = alpha;
                this.beta = beta;
                this.dim = dim; 
            }

            public final float alpha() { return alpha; }
            public final float beta() { return beta; }
            public final int[] dim() { return dim; }
                    
            @Override
            public final Tensor transform(Engine eg, byte[][] value) {
                return eg.tensor(alpha, value, beta, dim);
            }
        }
        //</editor-fold>
        public Int8ToTensor int8_to_tensor(float alpha, float beta, int[] dim) { return new Int8ToTensor(alpha, beta, dim); }
        public Int8ToTensor int8_to_tensor(int... dim) { return new Int8ToTensor(1.0f, 0.0f, dim); }
        
        //<editor-fold defaultstate="collapsed" desc="class: IntToTensor: byte[][] -> Tensor"> 
        public static class Uint8ToTensor implements Transform<byte[][]>
        {
            protected final int[] dim;
            protected final float alpha;
            protected final float beta;
            
            public Uint8ToTensor(float alpha, float beta, int... dim) {
                this.alpha = alpha;
                this.beta = beta;
                this.dim = dim; 
            }

            public final float alpha() { return alpha; }
            public final float beta() { return beta; }
            public final int[] dim() { return dim; }
                    
            @Override
            public final Tensor transform(Engine eg, byte[][] value) {
                return eg.img.tensor(alpha, value, beta, dim);
            }
        }
        //</editor-fold>
        public Uint8ToTensor uint8_to_tensor(float alpha, float beta, int[] dim) { return new Uint8ToTensor(alpha, beta, dim); }
        public Uint8ToTensor uint8_to_tensor(int... dim) { return new Uint8ToTensor(1.0f, 0.0f, dim); }
        
        //<editor-fold defaultstate="collapsed" desc="class: IntToTensor: byte[][] -> Tensor<int8>"> 
        public static class Int8ToTensorInt8 implements Transform<byte[][]>
        {
            protected final int[] dim;
            
            public Int8ToTensorInt8(int... dim) { this.dim = dim; }

            public final int[] dim() { return dim; }
                    
            @Override
            public final Tensor transform(Engine eg, byte[][] value) {
                return eg.tensor_int8(value, dim);
            }
        }
        //</editor-fold>
        public Int8ToTensorInt8 int8_to_tensor_int8(int... dim) { return new Int8ToTensorInt8(dim); }
        
        //<editor-fold defaultstate="collapsed" desc="class: OnehotTransform: int[] -> Tensor"> 
        public static class Onehot implements Transform<Integer[]> 
        {
            protected final int num_class;
            
            private Onehot(int num_class) {  this.num_class = num_class; }
            
            public final int num_class() { return num_class; }

            @Override
            public final Tensor transform(Engine eg, Integer[] value) {
                int[] labels = new int[value.length];
                for(int i=0; i<labels.length; i++) labels[i] = value[i];
                return eg.onehot(labels, num_class);
            }
        }
        //</editor-fold>
        public Onehot onehot(int num_class) { return new Onehot(num_class); }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="create: Buffer">
        public<T> Buffer<T> buffer(Callable<T> getter) { return new Buffer<>(getter); }
        
        public<K, V> Buffer<TensorPair> pair_buffer(Callable<TensorPair> getter) {
            return new Buffer<>(getter);
        }
        //</editor-fold>
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: States">
    public static class Stats
    {
        protected Stats() {}
        
        public static final Stats stat = new Stats();
        
        public static boolean default_partial = false;
        public static Charset default_charset = Charset.forName("UTF-8");
        
        //<editor-fold defaultstate="collapsed" desc="basic primitives">
        public boolean update(Stateful st, State dic) { return update(st, dic, default_partial); }
        public boolean update(Stateful st, State dic, boolean partial) {
            try { st.update_state(dic, partial); } 
            catch(Exception e) { throw new RuntimeException(e); }
            return true;
        }
        
        public Stateful transform(Stateful st, StatefulTransformer trf) {
            try { return trf.transform(st); }
            catch(Exception e) { throw new RuntimeException(e); }
        }
        
        public State read(StateReader writer) {
           try { return writer.read(); }
           catch(Exception e) { throw new RuntimeException(e); }
        }
        
        public boolean write(Stateful st, StateWriter writer) {
            try { writer.write(st.state()); }
            catch(Exception e) { throw new RuntimeException(e); }
            return true;
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="ZipState: extended">
        public ZipStateWriter zip_writer(String path) { return zip_writer(path, default_charset); }
        public ZipStateWriter zip_writer(String path, Charset cs) {
           try { return new ZipStateWriter(path, cs); }
           catch(IOException e) { throw new RuntimeException(e); }
        }
        
        public ZipStateReader zip_reader(String path) { return zip_reader(path, default_charset); } 
        public ZipStateReader zip_reader(String path, Charset cs) {
            try { return new ZipStateReader(path, cs); }
            catch(IOException e) { throw new RuntimeException(e); }
        }
        
        public boolean save_zip(Stateful st, String path) { return save_zip(st, path, default_charset); }
        public boolean save_zip(Stateful st, String path, Charset cs) {
            return write(st, zip_writer(path, cs));
        }
        
        public boolean load_zip(Stateful st, String path) {
            return load_zip(st, path, default_charset, default_partial);
        }
        public boolean load_zip(Stateful st, String path, boolean partial) { 
            return load_zip(st, path, default_charset, partial);
        }
        public boolean load_zip(Stateful st, String path, Charset cs, boolean partial) {//load and update
            State dic = zip_reader(path, cs).read();
            return update(st, dic, partial);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
