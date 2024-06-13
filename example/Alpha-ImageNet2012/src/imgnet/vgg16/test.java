/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.vgg16;

import static imgnet.Config.ImgNet2012_trainset;
import static imgnet.Config.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.Pair;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;

/**
 *
 * @author Gilgamesh
 */
public class test 
{
    static int batch_size = 256;
    
    public static void test() {
        VGG16 net = new VGG16().eval().init(eg).println();
        alpha.stat.load_zip(net, VGG16.weight);
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();
       
       BufferedTensorIter iter = ImgNet2012_trainset.init().shuffle_sort().buffered_iter(eg, batch_size);
        
        int batchIdx = 0;
        double accuracy = 0;
        eg.sync(false).check(false);
        
        for(; iter.hasNext(); batchIdx++) {
            Pair<Tensor, Tensor> pair = iter.next();
            Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
            Tensor Y = pair.label;
            
            Tensor Yh = net.forward(X)[0];
            
            System.out.println(batchIdx + ": loss = " + loss.loss(Yh, Y).get());
            
            Tensor predict = eg.row_max_index(Yh);//<int 32>
            Tensor label = eg.row_max_index(pair.label);//<int 32>
            
//            Vector.println("predict: ", predict.value_int32());
//            Vector.println("label: ", label.value_int32());
            
            float eq = eg.straight_equal(predict.c(), label.c()).get();
            System.out.println(batchIdx + ": eq = " + eq);
            accuracy += eq;
            
            net.gc(); predict.delete(); label.delete(); pair.label.delete();
        }
        
        System.out.println("accuracy = " + accuracy / batchIdx);
    }
    
    public static void main(String[] args) {
        try { test(); }
        catch(Exception e) { e.printStackTrace(); }
    }
}
