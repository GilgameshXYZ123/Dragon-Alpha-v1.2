/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool.adaptive;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class CoreAdaptiveMaxPool1D <T extends SimpleUnit> extends CoreAdaptivePool1D<T> {
    protected boolean training;
    transient protected Tensor Index;//Tensor<int32>

    public CoreAdaptiveMaxPool1D(T unit, boolean training, int out_width) {
        super(unit, out_width);
        this.training = training;
    }

    public final boolean training() { return training; } 
    public Tensor Index() { return Index; }
    
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
        __adaptive__(X);
        if(!training) return eg.pool1D_max(X, FW, OW, sw, 0);

        Tensor[] result = eg.pool1D_max_indexed(X, FW, OW, sw, 0);
        Index = result[1];//training: compute Index for: X -> Y
        return result[0];//deltaY = result[0]
    }
    
    @Override
    protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor deltaX = eg.unpool1D_max_indexed(deltaY, Index, IW, FW, sw, 0);
        return (grad_inplace? 
                deltaX.dual(()-> { deltaY.delete(); Index.delete(); }) : 
                deltaX.dual(()-> { Index.delete(); }));
    }
    //</editor-fold>
}
