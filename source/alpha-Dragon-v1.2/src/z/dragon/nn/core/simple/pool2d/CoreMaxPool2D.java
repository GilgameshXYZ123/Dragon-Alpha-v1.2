/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool2d;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreMaxPool2D<T extends SimpleUnit> extends CorePool2D<T>
{
    protected boolean training = true;
    
    transient protected int IH, IW;
    transient protected Tensor Index;//Tensor<int32>
    
    public CoreMaxPool2D(T unit, boolean training,
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width,
            int padding_height, int padding_width, 
            int output_height, int output_width) 
    {
        super(unit, 
              kernel_height, kernel_width, 
              stride_height, stride_width, 
              padding_height, padding_width,
              output_height, output_width);
        this.training = training;
    }

    public final boolean training() { return training; }
    public final Tensor Index() { return Index; }
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    public void variables(Tensor.TensorSet set) {
        super.variables(set);
        set.add(Index);
    }

    @Override
    public void gc() { 
        super.gc(); 
        if(Index != null) { Index.delete(); Index = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Engine eg, Tensor X) {
        if(!training) return eg.pool2D_max(X, FH, FW, OH, OW, sh, sw, ph, pw);
            
        int[] dim = X.dim(); IH = dim[1]; IW = dim[2];//training: compute Index for: X -> Y
        Tensor[] result = eg.pool2D_max_indexed(X, FH, FW, OH, OW, sh, sw, ph, pw);
        Index = result[1];
        return result[0];//deltaY = result[0]
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY,
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.unpool2D_max_indexed(deltaY, Index, IH, IW, FH, FW, sh, sw, ph, pw);
        return (grad_inplace?
                deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                deltaX.dual(()-> { Index.delete(); }));
    }
    //</editor-fold>
}
