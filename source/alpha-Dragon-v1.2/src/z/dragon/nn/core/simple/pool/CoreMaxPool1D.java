/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreMaxPool1D<T extends SimpleUnit> extends CorePool1D<T> {
    protected boolean training = true;
    transient protected int IW;
    transient protected Tensor Index;//Tensor<int32>
    
    public CoreMaxPool1D(T unit, boolean training,
            int kernel_width, 
            int stride_width,
            int padding_width, 
            int output_width) 
    {
        super(unit, 
              kernel_width, 
              stride_width, 
              padding_width,
              output_width);
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
        if(!training) return eg.pool1D_max(X, FW, OW, sw, pw);
            
        IW = X.dim(-2);//[N, W, C]
        Tensor[] result = eg.pool1D_max_indexed(X, FW, OW, sw,pw);
        Index = result[1];
        return result[0];//deltaY = result[0]
    }

    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY,
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.unpool1D_max_indexed(deltaY, Index, IW, FW, sw, pw);
        return (grad_inplace?
                deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                deltaX.dual(()-> { Index.delete(); }));
    }
    //</editor-fold>
}
