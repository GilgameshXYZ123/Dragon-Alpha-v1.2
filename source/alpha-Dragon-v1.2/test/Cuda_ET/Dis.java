/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Cuda_ET;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author Gilgamesh
 */
public class Dis 
{
    public static void write(String fname) throws FileNotFoundException, IOException {
        File f = new File(fname);
        FileReader fr = new FileReader(f);
        BufferedReader bufr = new BufferedReader(fr);
        
        FileWriter fw = new FileWriter(new File(fname + "new"));
        BufferedWriter bufw = new BufferedWriter(fw);
        
        int len = 1605;
        int[] value = new int[len];
        
        String line = null;
        while((line = bufr.readLine()) != null) {
            String[] tokens = line.split(", ");
            int error = Integer.valueOf(tokens[0]);
            int count = Integer.valueOf(tokens[1]);
            value[error] = count;
        }
        
        for(int i=0; i<len; i++) {
            System.out.println(value[i]);
            bufw.write(i + ", " + value[i] + "\n");
        }
        
        bufr.close();
        fr.close();
        
        bufw.close();
        fw.close();
    }
    
    public static void main(String[] args) throws IOException
    {
        write("F:\\Gilgamesh-book-SA\\z-project\\[5] efficient and flexible Winograd Convolution for NHWC layout"
                + "\\datas\\distribution\\F(10, 7)\\txt\\Im2col-Winograd-F(10, 7).txt");
    }
}
