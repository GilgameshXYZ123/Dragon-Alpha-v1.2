/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Raw;

/**
 *
 * @author Gilgamesh
 */
public class ImgSize 
{
    public static void main(String[] args)
    {
        int FH = 9, FW = 9;
	int N = 128, OH = 56, OW = 56, IC = 64, OC = 64;
        //int N = 64, OH = 128, OW = 128, IC = 64, OC = 64;
	
	int IH = OH, IW = OW;
	if(FH % 2 == 0) IH--;
	if(FW % 2 == 0) IW--;
	
	double Wsize = IC * FH * FW * OC;
	double Xsize = N * IH * IW * IC;
	double Ysize = N * OH * OW * OC;
	
	Xsize = Xsize * 4 / 1024 / 1024;
	Ysize = Ysize * 4 / 1024 / 1024; 
	Wsize = Wsize * 4 / 1024 / 1024;
	
        System.out.println("W: " + Wsize);
        System.out.println("X: " + Xsize);
        System.out.println("Y: " + Ysize);
    }
    
}
