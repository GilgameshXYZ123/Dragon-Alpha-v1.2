/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.spring;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.unit.simple.math1.Abs;
import z.dragon.nn.unit.simple.math1.Cos;
import z.dragon.nn.unit.simple.math1.Quadratic;
import z.dragon.nn.unit.simple.math1.Sin;
import z.dragon.nn.unit.simple.math2.Arcsin;
import z.dragon.nn.unit.simple.math2.Arctan;
import z.dragon.nn.unit.simple.math2.Cot;
import z.dragon.nn.unit.simple.math2.Elu;
import z.dragon.nn.unit.simple.math2.Exp;
import z.dragon.nn.unit.simple.math2.LeakyRelu;
import z.dragon.nn.unit.simple.math2.Linear;
import z.dragon.nn.unit.simple.math2.Log;
import z.dragon.nn.unit.simple.math2.LogSoftmax;
import z.dragon.nn.unit.simple.math2.Relu;
import z.dragon.nn.unit.simple.math2.Rpl;
import z.dragon.nn.unit.simple.math2.Sigmoid;
import z.dragon.nn.unit.simple.math2.Softmax;
import z.dragon.nn.unit.simple.math2.Softplus;
import z.dragon.nn.unit.simple.math2.Sqrt;
import z.dragon.nn.unit.simple.math2.Tan;
import z.dragon.nn.unit.simple.math2.Tanh;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Cuda_function_test5 
{
    static final ExRandom exr = new ExRandom();
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha");}
    static Engine eg = alpha.engine.cuda_float32(0, alpha.engine.memp1());
    
    static float[][] mX = new float[][]{//7 * 3
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f,  0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f,  0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
        { 0.1f, 0.2f,  0.3f, 0.4f,  0.3f, -0.2f, -0.1f},
        {-0.5f, 0.5f, -0.4f, 0.5f, -0.2f,  0.9f,  0.2f},
        { 0.1f, 0.1f,  0.2f, 0.6f,  0.1f,  0.1f, -0.8f},
        {-0.1f, 0.2f, -0.3f, 0.4f, -0.3f,  0.2f, -0.1f},
        { 0.5f, 0.5f,  0.4f, 0.5f,  0.2f,  0.9f, -0.2f},
        {-0.1f, 0.1f, -0.2f, 0.6f, -0.1f, -0.1f,  0.8f},
    };
    
    static int height = mX.length;
    static int width = mX[0].length;

    static Tensor tX = eg.tensor(mX, height, width);
        
    public static void test2()
    {
        //test forward prop-----------------------------------------------------
//        Relu relu = nn.relu().init(eg);
//        Tensor tY = relu.forward(tX)[0];
//        Tensor tdeltaX = relu.backward(eg.ones_like(tY).c())[0];
        
//        LeakyRelu leakyRelu = nn.leakyRelu().init(eg);
//        Tensor tY = leakyRelu.forward(tX)[0];
//        Tensor tdeltaX = leakyRelu.backward(eg.ones_like(tY).c())[0];
        
//        Softplus softplus = nn.softplus().init(eg);
//        Tensor tY = softplus.forward(tX)[0];
//        Tensor tdeltaX = softplus.backward(eg.ones_like(tY).c())[0];
        
//        Elu elu = nn.elu().init(eg);
//        Tensor tY = elu.forward(tX)[0];
//        Tensor tdeltaX = elu.backward(eg.ones_like(tY).c())[0];
        
        //======================================================================
//        Exp exp = nn.exp(2.0f, 0.5f).init(eg);
//        Tensor tY = exp.forward(tX)[0];
//        Tensor tdeltaX = exp.backward(eg.ones_like(tY))[0];
        
//        Log log = nn.log(0.5f, 0.5f).init(eg);
//        Tensor tY = log.forward(tX)[0];
//        Tensor tdeltaX = log.backward(eg.ones_like(tY))[0];
        
//        Sqrt sqrt = nn.sqrt(2.0f, 5f).init(eg);
//        Tensor tY = sqrt.forward(tX)[0];
//        Tensor tdeltaX = sqrt.backward(eg.ones_like(tY))[0];
        
//        Softmax softmax = nn.softmax(false).init(eg);
//        Tensor tY = softmax.forward(tX)[0];
//        Tensor tdeltaX = softmax.backward(tdeltaY)[0];
        
        LogSoftmax log_softmax = nn.log_softmax().init(eg);
        Tensor tY = log_softmax.forward(tX)[0];
        Tensor tdeltaX = log_softmax.backward(eg.ones_like(tY))[0];
        
//        Sigmoid sigmoid = nn.sigmoid().init(eg);
//        Tensor tY = sigmoid.forward(tX)[0];
//        Tensor tdeltaX = sigmoid.backward(eg.ones_like(tY))[0];
        
//        Tanh tanh = nn.tanh().init(eg);
//        Tensor tY = tanh.forward(tX)[0];
//        Tensor tdeltaX = tanh.backward(eg.ones_like(tY))[0];
        
        //======================================================================
//        Linear linear = nn.linear(2.0f, 0.5f).init(eg);
//        Tensor tY = linear.forward(tX)[0];
//        Tensor tdeltaX = linear.backward(eg.ones_like(tY))[0];
        
//        Rpl rpl = nn.rpl().init(eg);
//        Tensor tY = rpl.forward(tX)[0];
//        Tensor tdeltaX = rpl.backward(eg.ones_like(tY))[0];
        
//        Quadratic qua = nn.quadratic(3, 2, 1).init(eg);
//        Tensor tY = qua.forward(tX)[0];
//        Tensor tdeltaX = qua.backward(eg.ones_like(tY))[0];
        
//        Abs abs = nn.abs(2, 0.5f).init(eg);
//        Tensor tY = abs.forward(tX)[0];
//        Tensor tdeltaX = abs.backward(eg.ones_like(tY))[0];
      
        //======================================================================
//        Tan tan = nn.tan(2.0f, 0.5f).init(eg);
//        Tensor tY = tan.forward(tX)[0];
//        Tensor tdeltaX = tan.backward(eg.ones_like(tY))[0];
        
//        Cot cot = nn.cot(2.0f, 0.5f).init(eg);
//        Tensor tY = cot.forward(tX)[0];
//        Tensor tdeltaX = cot.backward(eg.ones_like(tY))[0];
        
//        Sin sin = nn.sin(2.0f, 0.5f).init(eg);
//        Tensor tY = sin.forward(tX)[0];
//        Tensor tdeltaX = sin.backward(eg.ones_like(tY))[0];
        
//        Cos cos = nn.cos(2.0f, 0.5f).init(eg);
//        Tensor tY = cos.forward(tX)[0];
//        Tensor tdeltaX = cos.backward(eg.ones_like(tY))[0];
        
//        Arcsin asin = nn.arcsin(2.0f, 0.5f).init(eg);
//        Tensor tY = asin.forward(tX)[0];
//        Tensor tdeltaX = asin.backward(eg.ones_like(tY))[0];
        
//        Arctan atan = nn.arctan(2.0f, 0.5f).init(eg);
//        Tensor tY = atan.forward(tX)[0];
//        Tensor tdeltaX = atan.backward(eg.ones_like(tY))[0];
    
        
        //forward---------------------------------------------------------------
        float sum1 = eg.straight_sum(tY).get();
        System.out.println("sum = " + sum1);
        
        float[] Y = tY.value();
        float[][] mY = Vector.to2D(Y, height, width);
        Matrix.println(mY);
        
        //backward--------------------------------------------------------------
        float sum2 = eg.straight_sum(tdeltaX).get();
        System.out.println("sum2 = " + sum2);
        
        float[] deltaX = tdeltaX.value();
        float[][] mdeltaX = Vector.to2D(deltaX, height, width);
        Matrix.println(mdeltaX);
        
        if(Float.isNaN(sum1)) throw new NullPointerException();
        if(Float.isNaN(sum2)) throw new NullPointerException();
    }
    
    public static void main(String[] args)
    {
        test2();
    }
}
