/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_relu;

import cifar10.resnet18_relu.Net.ResNet18;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.data.TensorIter;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
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
    
    static int batch_size = 64;
    public static void test() throws IOException
    {
        ResNet18 net = new ResNet18().init(eg).println(); net.load();
        net.eval();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
//        DataSet<byte[], Integer> dataset = Cifar10.train();
        DataSet<byte[], Integer> dataset = Cifar10.test();
//        
//        eg.sync(false).check(false);
        double accuracy = 0;
        int batchIdx = 0;
        TensorIter iter = dataset.batch_iter();
        while(iter.hasNext()) 
        {
            Pair<Tensor, Tensor> pair = iter.next(eg, batch_size);
            Tensor x = pair.input.c().linear(true, 2.0f, -1.0f);
            Tensor y = pair.label;
            
            Tensor yh = net.forward(x)[0];
            
            float ls = loss.loss(yh, y).get();
            System.out.println("loss = " + ls);
            
            Tensor pre  = eg.row_max_index(yh).c();//<int 32>
            Tensor real = eg.row_max_index(pair.label).c();
            Vector.println("Y1: ", pre.value_int32(), 0, batch_size);
            Vector.println("Y2: ", real.value_int32(), 0, batch_size);
            float eq = eg.straight_equal(pre, real).get();
            accuracy += eq;
            
            System.out.println("eq = " + eq);
            
            net.gc(); pre.delete(); real.delete(); 
            batchIdx++;
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
