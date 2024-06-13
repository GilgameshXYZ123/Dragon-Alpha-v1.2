/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paper.net_present;

import static z.dragon.alpha.Alpha.Loss.loss;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;

/**
 *
 * @author Gilgamesh
 */
public class Fused 
{
    float negative_slope;
    boolean inplace;
    int features;
    int[] feature_dim;
    Tensor X1, X2;
    float nonzero_prop;
    float brightness;
    float saturation;
    float contrast;
    float theta;
    float shy, shx;
    float ty, tx;
    int height, width;
    
    Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    public void train() 
    {
        nn.batchNorm_leakyRelu(nn.batchNorm(inplace, feature_dim), nn.leakyRelu(negative_slope));
        nn.batchNorm_relu(nn.batchNorm(feature_dim), nn.relu(inplace));
        nn.global_batchNorm_leakyRelu(nn.global_batchNorm(feature_dim), nn.leakyRelu());
        nn.affine_leakyRelu(nn.affine(inplace, feature_dim), nn.leakyRelu());
        nn.leakyRelu_dropout(nn.leakyRelu(inplace, negative_slope), nn.dropout(inplace, nonzero_prop));
        
        F.add_leakyRelu(negative_slope, X1, X2);
        F.log_softmax(inplace, X1);
        
        loss.softmax_crossEntropy(features);
        loss.sigmoid_binaryCrossEntropy();
        
        eg.img.adjust_color(inplace, X1.to_int8(inplace), brightness, saturation, contrast);
        eg.img.affine().rotate(theta).shear(shy, shx).translate(ty, tx)
                .transform(X1, inplace, height, width);
    }
}
