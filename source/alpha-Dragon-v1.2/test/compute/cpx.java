/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package compute;

/**
 *
 * @author Gilgamesh
 */
public class cpx 
{
    static int complex(int IH, int IW, int FH, int FW, int sh, int sw, int ph, int pw) {
        int OH = (IH + 2*ph - FH)/sh + 1;
        int OW = (IW + 2*pw - FW)/sw + 1;
        return OH * OW * FH * FW;
    }
    
    static int complex_V2(int IH, int IW, int FH, int FW, int sh, int sw, int ph, int pw) {
        int OH = (IH + 2*ph - FH)/sh + 1;
        int OW = (IW + 2*pw - FW)/sw + 1;
        
        int sum = 0;
        for(int oh=0; oh<OH; oh++)
        for(int ow=0; ow<OW; ow++) {
            int toh = oh*sh - ph;
            int tow = ow*sw - pw;
            int fhs = -(toh<0? toh : 0);
            int fws = -(tow<0? tow : 0);
            int fhe = IH - toh; fhe = (fhe > FH? FH : fhe);
            int fwe = IW - tow; fwe = (fwe > FW? FW : fwe);
            int k = (fhe - fhs) * (fwe - fws);
            sum += k;
        }
        return sum;
    }
    
    static float pc(int IH, int IW, int FH, int FW, int ph, int pw, int sh, int sw) {
        int cpx1 = complex(IH, IW, FH, FW, sh, sw, ph, pw);
        int cpx2 = complex_V2(IH, IW, FH, FW, sh, sw, ph, pw);
        return 1.0f * (cpx1 - cpx2) / cpx1;
    }
    
    static float p0(int IH, int IW, int FH, int FW, int ph, int pw, int sh, int sw) {
        int OH = (IH + 2*ph - FH)/sh + 1;
        int OW = (IW + 2*pw - FW)/sw + 1;
        int IHp = (OH - 1)*sh + FH;
        int IWp = (OW - 1)*sw + FW;
        return 1.0f - 1.0f*(IH*IW)/(IHp*IWp);
    }

    public static void main(String[] args)
    {
        int IH = 10, IW = 10;
        int FH = 3, FW = 3;
        int sh = 2, sw = 2;
        int ph = 1, pw = 1;
        
        float pc = pc(IH, IW, FH, FW, ph, pw, sh, sw);
        float p0 = p0(IH, IW, FH, FW, ph, pw, sh, sw);
        
//        System.out.println("pc = " + pc);
//        System.out.println("pc0 = " + p0);
//        System.out.println("pc / p0 = " + pc / p0);
        
        for(int i=32; i>=2; i-=2) {
            float a = pc(i, i, FH, FW, ph, pw, sh, sw);
            float b = p0(i, i, FH, FW, ph, pw, sh, sw);
            System.out.println(i + " " + a + " " +  b);
        }
    }
    
}
