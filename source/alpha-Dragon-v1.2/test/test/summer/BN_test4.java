/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.cuda.summer;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.dragon.nn.unit.complex.Module;
import z.dragon.nn.unit.simple.affine.BatchNorm;
import z.dragon.nn.unit.simple.affine.SqBatchNorm;

/**
 *
 * @author Gilgamesh
 */
public class BN_test4 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    static {
        CudaFloat32EngineBase cu32  = (CudaFloat32EngineBase) eg.engineBase();
//        cu32.field_var_f64(true);
    }
    
    public static float[] random1D(int length, float min, float max) {
        if(min > max) { float t = min; min = max; max = t;}
        float div = max - min;
        
        int seed = 123;
        float[] X = new float[length];
        for(int i=0; i<length; i++) {
            seed = ((632229*seed) + 2100473) & 4194303;
            float v = min + div * (seed / 4194304.0f);
            X[i] = v;
        }
        return X;
    }
    
    public static float[][] random2D(int height, int width, float min, float max) {
        if(min > max) { float t = min; min = max; max = t;}
        float div = max - min;
        
        int seed = 123;
        float[][] X = new float[height][width];
        for(int i=0; i<height; i++)
            for(int j=0; j< width; j++) {
                seed = ((632229*seed) + 2100473) & 4194303;
                float v = min + div * (seed / 4194304.0f);
                X[i][j] = v;
            }
        
        return X;
    }
    
    public static void test1(int height, int width)
    {
        //prepare Area----------------------------------------------------------
        float[][] X = random2D(height, width, 0, 1);
        float[][] L = random2D(height, width, 0, 1);
        
        //forward prop----------------------------------------------------------
        float eps = 1e-5f;
        boolean inplace = false;
        BatchNorm bn0 = nn.batchNorm(inplace, width).eps(eps);
        BatchNorm bn1 = nn.batchNorm(inplace, width).eps(eps);  
        BatchNorm bn2 = nn.batchNorm(inplace, width).eps(eps); 
        BatchNorm bn3 = nn.batchNorm(inplace, width).eps(eps);

//        SqBatchNorm bn0 = nn.sqBatchNorm(inplace, width).eps(eps);
//        SqBatchNorm bn1 = nn.sqBatchNorm(inplace, width).eps(eps);  
//        SqBatchNorm bn2 = nn.sqBatchNorm(inplace, width).eps(eps); 
//        SqBatchNorm bn3 = nn.sqBatchNorm(inplace, width).eps(eps);

//        bn0.hook_after_backward((self)->{ alpha.print(bn0.deltaX().data(), '\n'); }); 
//        bn1.hook_after_backward((self)->{ alpha.print(bn1.deltaX().data(), '\n'); }); 
//        bn2.hook_after_backward((self)->{ alpha.print(bn2.deltaX().data(), '\n'); }); 
//        bn3.hook_after_backward((self)->{ alpha.print(bn3.deltaX().data(), '\n'); }); 
        
        float k = 1;
        Module net = new Module() {
            @Override
            public Tensor[] __forward__(Tensor... X) {
                X = bn0.forward(X);
                X = F.leakyRelu(false, k, X);
//                X = F.relu(false, X);
//                X = F.sigmoid(false, X);
//                alpha.print(X[0].v(), '\n');

                X = bn1.forward(X);
                X = F.leakyRelu(false, k, X);
//                X = F.relu(false, X);
//                X = F.sigmoid(false, X);
//                alpha.print(X[0].v(), '\n');
                
                X = bn2.forward(X);
                X = F.leakyRelu(false, k, X);
//                X = F.relu(false, X);
//                X = F.sigmoid(false, X);
//                alpha.print(X[0].v(), '\n');
                
                X = bn3.forward(X);
                X = F.leakyRelu(false,  k, X);
//                X = F.relu(false, X);
//                X = F.sigmoid(false, X);
//                alpha.print(X[0].v());
                
                return X;
            }
        }.init(eg);
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        Optimizer opt = alpha.optim.Adam(net.params(), 0.01f);
        
        Tensor tA0 = bn0.weight(), tB0 = bn0.bias();
        Tensor tA1 = bn1.weight(), tB1 = bn1.bias();
        Tensor tA2 = bn2.weight(), tB2 = bn2.bias();
        Tensor tA3 = bn3.weight(), tB3 = bn3.bias();
        
        //backward prop----------------------------------------------------------
        for(int i=0; i<10; i++) {
            System.out.println("\n\nround = " + i);
            
            Tensor tX = eg.tensor(X, width).need_grad(true);
            Tensor tL = eg.tensor(L, width);
            
            Tensor tY = net.forward(tX)[0];
            System.out.println("tY.sum = " + tY.sum());
            alpha.print(tY.data());
            
            Tensor tdeltaX =  net.backward(loss.gradient(tY, tL))[0]; 
            System.out.println("tdeltaX.sum" + tdeltaX.sum());
            alpha.print(tdeltaX.data());
            
            //check grads-------------------------------------------------------
            Tensor dA0 = tA0.grad();
            Tensor dA1 = tA1.grad();
            Tensor dA2 = tA2.grad();
            Tensor dA3 = tA3.grad();
            
//            if(i == 0) {
//                dA0.set(new float[]{1.8350e-06f,  1.4218e-06f,  2.5331e-06f, -6.1117e-07f});
//                dA1.set(new float[]{-1.3262e-06f,  1.2630e-07f,  5.7018e-07f, -2.2272e-07f});
//                dA2.set(new float[]{1.8915e-06f, -7.8963e-07f,  3.4890e-07f,  9.7806e-07f});
//            }
            
            if(dA0 != null) alpha.print("dA0.sum = ", dA0.sum(), dA0.value());
            if(dA1 != null) alpha.print("dA1.sum = ", dA1.sum(), dA1.value());
            if(dA2 != null) alpha.print("dA2.sum = ", dA2.sum(), dA2.value());
            if(dA3 != null) alpha.print("dA3.sum = ", dA3.sum(), dA3.value());
            System.out.println();
            
            //check grads-------------------------------------------------------
            opt.update();
            alpha.print("tA0.sum = ", tA0.sum(), tA0.value());
            alpha.print("tA1.sum = ", tA1.sum(), tA1.value());
            alpha.print("tA2.sum = ", tA2.sum(), tA2.value());
            alpha.print("tA3.sum = ", tA3.sum(), tA3.value());
            System.out.println();
            
            opt.clear_grads();
            net.gc();
        }
    }
    
    public static void main(String[] args)
    {
        try
        {
            test1(8, 4);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
    }
    
}
