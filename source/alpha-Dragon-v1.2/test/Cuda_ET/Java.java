/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import static z.dragon.alpha.Alpha.alpha;

/**
 *
 * @author Gilgamesh
 */
public class Java 
{
    public static void repair() {
        byte[] bytes1 = alpha.fl.to_bytes("C:\\Users\\Gilgamesh\\Desktop\\main1.tex");
        byte[] bytes2 = alpha.fl.to_bytes("C:\\Users\\Gilgamesh\\Desktop\\main.tex");
//        for (int i=0; i<bytes2.length; i++) {
//            if (bytes1[i] == bytes2[i]) System.out.println(i);
//        }
        
        for(int i=0; i<1000; i++) bytes1[i] = bytes2[i];
         
        alpha.fl.wt_bytes("C:\\Users\\Gilgamesh\\Desktop\\main2.tex", bytes1);
        
        System.out.println(bytes1.length);
    }
    
    
    public static void main(String[] args) {
        repair();
    }
}
