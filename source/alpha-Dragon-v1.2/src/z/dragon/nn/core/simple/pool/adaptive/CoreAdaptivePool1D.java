/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool.adaptive;

import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class CoreAdaptivePool1D<T extends SimpleUnit> extends SimpleCore<T> {
    protected int OW;
    
    transient protected int FW;
    transient protected int sw;
    transient protected int IW;
    
    protected CoreAdaptivePool1D(T unit, int out_width)  {
        super(unit);
        if(out_width <= 0) throw new IllegalArgumentException("out_width <= 0");
        this.OW = out_width;
    }
    
    public final int[] kernel()   { return new int[]{ FW }; }
    public final int[] stride()   { return new int[]{ sw }; }
    public final int[] padding()  { return new int[]{ 0 }; }
    public final int[] out_size() { return new int[]{ OW };}
    
    protected void __adaptive__(Tensor X) {//X[N, IW, IC]
        IW = X.dim(-2); 
        sw = Math.floorDiv(IW, OW);
        FW = IW - (OW - 1)*sw;
    }
}
