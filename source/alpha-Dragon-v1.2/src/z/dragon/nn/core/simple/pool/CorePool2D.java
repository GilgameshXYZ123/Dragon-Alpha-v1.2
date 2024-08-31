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
public abstract class CorePool2D<T extends SimpleUnit> extends SimpleCore<T> {
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;

    public CorePool2D(T unit, 
            int kernel_height,  int kernel_width, 
            int stride_height,  int stride_width,
            int padding_height, int padding_width,
            int output_height,  int output_width) 
    { 
        super(unit);
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }
    
    public final int[] kernel()   { return new int[]{ FH, FW }; }
    public final int[] stride()   { return new int[]{ sh, sw }; }
    public final int[] padding()  { return new int[]{ ph, pw }; }
    public final int[] out_size() { return new int[]{ OH, OW }; }
}
