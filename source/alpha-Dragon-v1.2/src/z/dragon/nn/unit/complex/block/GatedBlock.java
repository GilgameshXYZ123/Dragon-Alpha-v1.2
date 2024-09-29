/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.block;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class GatedBlock extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit gate_fc, conv_fc, idty_fc;
    public Unit out_fc, conv, split, cat, add, mul;
    public Unit act, norm;
    
    public GatedBlock(Unit activation, Unit normalize,
            int in_channel, float conv_ratio /*1.0*/, int expand,/*2*/
            int kernel/*7*/) 
    {
        act = activation;
        norm = normalize;
        
        int hidden       = in_channel * expand;
        int conv_channel = (int) (in_channel * conv_ratio);
        int idty_channel = hidden - conv_channel;
        
        gate_fc = nn.fullconnect(false, in_channel, hidden);
        conv_fc = nn.fullconnect(false, in_channel, conv_channel);
        if (idty_channel > 0) {
            idty_fc = nn.fullconnect(false, in_channel, idty_channel);
            cat = nn.concat(-1);
        }
        conv = nn.depthwise_conv3D(false, conv_channel, kernel, 1, kernel / 2);
        
        mul = nn.mul();
        out_fc = nn.fullconnect(false, hidden, in_channel);
        add = nn.add();
    }

    @Override
    public Tensor[] __forward__(Tensor... X) {
        Tensor res = X[0];
        if (norm != null) X = norm.forward(X);
        
        Tensor G = gate_fc.forward(X)[0];//[N, H, W, C] -> [N, H, W, hidden]
        Tensor C = conv_fc.forward(X)[0];//[N, H, W, C] -> [N, H, W, conv_channel]
        Tensor I = null; if (idty_fc != null) I = idty_fc.forward(X)[0];//[N, H, W, C] -> [N, H, W, idty_channel]
        
        G =  act.forward(G)[0];
        C = conv.forward(C)[0];
        if (I != null) C = cat.forward(C, I)[0];//[N, H, W, idty_channel + conv_channel] -> [N, H, W, hiddeen] 
        Tensor Y = out_fc.forward(mul.forward(G, C))[0];//Y = F(G * C)
        
        return add.forward(Y, res);//[N, H, W, hidden] -> [N, H, W, C] 
    }
}
