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
public class kernel_split
{
    public static void stage1(int FH, int FW, int sh, int sw) {
        for(int y=0; y<sh; y++)
        for(int x=0; x<sw; x++) {
            int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
            int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
            
            System.out.println("x = " + x + ", y = " + y + ": ");
            
            for(int fhr=0; fhr<CFH; fhr++) {
                for(int fwr=0; fwr<CFW; fwr++) {
                    int fh = y + (oph - fhr)*sh;
                    int fw = x + (opw - fwr)*sw;
                
                    System.out.print("fh, fw = " + fh + ", " + fw + "\t");
                }
                System.out.println();
            }
        }
    }
    
    public static void stage2(int IH, int IW, int FH, int FW, int sh, int sw, int ph, int pw) 
    {
        int IH_slice = (IH + sh - 1) / sh;
        int IW_slice = (IW + sw - 1) / sw;
                
        for(int y=0; y<sh; y++)
        for(int x=0; x<sw; x++) 
        {
            System.out.println("(y, x) = " + y + ", " + x);
            
            int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
            int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
             System.out.println("(oph, opw) = " + oph + ", " + opw);
           
            int ihs = (y - ph); if(ihs < 0) ihs += (ph - y + sh - 1)/sh*sh; //make sure: ihs, iws >= 0
            int iws = (x - pw); if(iws < 0) iws += (pw - x + sw - 1)/sw*sw;
            System.out.println("(ihs, iws) = " + ihs + ", " + iws);
            
            for(int u=0; u<IH_slice; u++)
            for(int v=0; v<IW_slice; v++)
            {
                int ih = u*sh + ihs;
                int iw = v*sw + iws;
                System.out.println("(ih, iw) =" + ih + ", " + iw);
                
                int ohs = (ih + ph - y)/sh - oph;
                int ows = (iw + pw - x)/sw - opw;
                System.out.println("(ohs, ows) =" + ohs + ", " + ows);
            }
            System.out.println();
        }
    }
    

    public static void main(String[] args) 
    {
        int IH = 4, IW = 4;
        int FH = 3, FW = 3;
        int sh = 2, sw = 2;
        int ph = 1, pw = 1;
//        stage1(FH, FW, sh, sw);
        stage2(IH, IW, FH, FW, sh, sw, ph, pw);
    }
}
