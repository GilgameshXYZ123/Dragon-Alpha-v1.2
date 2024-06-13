/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool2d;

import z.dragon.engine.Tensor;
import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class CoreAdaptivePool2D<T extends SimpleUnit> extends SimpleCore<T> 
{
    protected int OH, OW;
    
    transient protected int FH, FW;
    transient protected int sh, sw;
    transient protected int IH, IW;
    
    protected CoreAdaptivePool2D(T unit,
            int out_height, int out_width) 
    {
        super(unit);
        if(out_height <= 0) throw new IllegalArgumentException("out_height <= 0");
        if(out_width  <= 0) throw new IllegalArgumentException("out_width <= 0");
        this.OH = out_height;
        this.OW = out_width;
    }
    
    public final int[] kernel() { return new int[]{ FH, FW }; }
    public final int[] stride() { return new int[]{ sh, sw }; }
    public final int[] padding() { return new int[]{ 0, 0 }; }
    public final int[] out_size() {return new int[]{ OH, OW };}
    
    protected void __adaptive__(Tensor X) {
        int[] dim = X.dim();//X[N, IH, IW, IC]
        if(dim.length != 4) throw new IllegalArgumentException("Tensor X.ndim != 4");
            
        IH = dim[1]; sh = Math.floorDiv(IH, OH); FH = IH - (OH - 1)*sh;
        IW = dim[2]; sw = Math.floorDiv(IW, OW); FW = IW - (OW - 1)*sw;
    }
}
