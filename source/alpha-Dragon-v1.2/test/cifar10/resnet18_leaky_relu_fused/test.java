/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_leaky_relu_fused;

import cifar10.resnet18_leaky_relu_fused.Net.ResNet18;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.data.TensorIter;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.cuda.CudaFloat32EngineBase;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.unit.simple.blas.Conv3D;
import z.dragon.nn.unit.simple.blas.FullConnect;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp3(alpha.MEM_1GB * 6);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
   
    
    static int batch_size = 128;
    public static void test() {
        ResNet18 net = new ResNet18().eval().init(eg).println();
        net.load();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        DataSet<byte[], Integer> dataset = Cifar10.train();
//        DataSet<byte[], Integer> dataset = Cifar10.test();
//        
        int batchIdx = 0;
        double accuracy = 0;
        eg.sync(false).check(false);
        
        for(TensorIter iter = dataset.batch_iter(); iter.hasNext(); batchIdx++) {
            Pair<Tensor, Tensor> pair = iter.next(eg, batch_size);
            Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
            Tensor Y = pair.label;
            
            Tensor Yh = net.forward(X)[0];
            
            System.out.println("loss = " + loss.loss(Yh, Y).get());
            
            Tensor pre  = eg.row_max_index(Yh);
            Tensor real = eg.row_max_index(pair.label);
            
            Vector.println("Y1: ", pre.value_int32());
            Vector.println("Y2: ", real.value_int32());
            
            float eq = eg.straight_equal(pre, real).get();
            System.out.println("eq = " + eq);
            accuracy += eq;
            
            net.gc(); pre.delete(); real.delete(); 
        }
        
        System.out.println("accuracy = " + accuracy / batchIdx);
    }
    
    @SuppressWarnings("unchecked")
    public static void main(String[] args)
    {
        try
        {
            test();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
