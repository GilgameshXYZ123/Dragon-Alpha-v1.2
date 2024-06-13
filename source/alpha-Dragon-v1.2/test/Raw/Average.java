/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Raw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import static z.dragon.alpha.Alpha.alpha;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Average 
{
    public static void average() throws FileNotFoundException, IOException 
    {
        File f = new File("C:\\Users\\Gilgamesh\\Desktop\\k.txt");
        FileReader fr = new FileReader(f);
        BufferedReader bufr = new BufferedReader(fr);
        
        float[] k1 = new float[3];
        float[] k2 = new float[3];
        
        int index1 = 0, index2 = 0;
        String line;
        while(true) {
            line = bufr.readLine(); if(line == null) break;
            k1[index1++] = Float.valueOf(line);
            k2[index2++] = Float.valueOf(bufr.readLine());
        }
        
        bufr.close();
        fr.close();
        
        System.out.println(Vector.average(k1));
        System.out.println(Vector.average(k2));
    }
    
    public static void main(String[] args) throws IOException
    {
        average();
    }
    
}
