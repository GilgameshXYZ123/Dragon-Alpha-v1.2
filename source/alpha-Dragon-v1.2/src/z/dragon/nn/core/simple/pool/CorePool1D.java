/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.core.simple.pool;

import z.dragon.nn.core.simple.SimpleCore;
import z.dragon.nn.unit.simple.SimpleUnit;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public abstract class CorePool1D<T extends SimpleUnit> extends SimpleCore<T> {
    protected int FW;
    protected int sw;
    protected int pw;
    protected int OW;

    public CorePool1D(T unit, 
            int kernel_width, 
            int stride_width,
            int padding_width,
            int output_width) 
    { 
        super(unit);
        this.FW = kernel_width;
        this.sw = stride_width;
        this.pw = padding_width;
        this.OW = output_width;
    }
    
    public final int[] kernel()   { return new int[]{ FW }; }
    public final int[] stride()   { return new int[]{ sw }; }
    public final int[] padding()  { return new int[]{ pw }; }
    public final int[] out_size() { return new int[]{ OW }; }
}
