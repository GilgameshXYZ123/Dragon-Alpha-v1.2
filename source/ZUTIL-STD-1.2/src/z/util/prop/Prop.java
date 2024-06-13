/*
 * To change this license header, choose License Headers in Project Prop.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.prop;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import z.util.lang.Resources;

/**
 *
 * @author dell
 */
public class Prop 
{
    public static Properties getProp(String src)
    {
        Properties prop=null;
        InputStream in=null;
        try
        {
            in=Resources.getResourceAsStream(src);
            prop=new Properties();
            prop.load(in);
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            try
            {
                if(in!=null) in.close();
            }
            catch(Exception e)
            {
                System.err.println(e);
                throw new RuntimeException(e);
            }
        }
        return prop;
    }
    public static Properties getProp(InputStream in) throws IOException
    {
        Properties prop=new Properties();
        prop.load(in);
        return prop;
    }
    public static Properties getProp(byte[] buf) 
    {
        Properties prop=null;
        InputStream in=null;
        try
        {
            in=new ByteArrayInputStream(buf);
            prop=new Properties();
            prop.load(in);
        }
        catch(IOException e)
        {
            System.err.println(e);
            throw new RuntimeException();
        }
        return prop;
    }
}
