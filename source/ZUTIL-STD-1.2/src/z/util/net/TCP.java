/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.net;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
/**
 *
 * @author dell
 */
public class TCP 
{
    //<editor-fold defaultstate="collapsed" desc="class TCPService">
    public static class TCPService implements Runnable
    {
        //columns-------------------------------------------------------------------
        protected Socket client=null;
    
        //constructor---------------------------------------------------------------
        public TCPService(Socket client)    
        {
            if(client==null) throw new NullPointerException();
            this.client=client;
        }
        //functions-----------------------------------------------------------------
        @Override
        public void run() 
        {
            InputStream in=null;
            OutputStream out=null;
            try
            {
                in=client.getInputStream();
                byte[] buf=new byte[1024];
                int len=in.read(buf);
                if(len!=-1)
                {
                    String msg=Thread.currentThread().getId()+"Received"+new String(buf,0,len);
                    System.out.println(msg);
                    out=client.getOutputStream();
                    out.write(msg.getBytes());
                }
            }
            catch(IOException e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
            finally
            {
                try
                {
                    client.close();
                    if(in!=null) in.close();
                }
                catch(IOException e)
                {
                    e.printStackTrace();
                    throw new RuntimeException(e);
                }
            }
        }
    }
    //</editor-fold>
    public static ServerSocket newServer(String host, int port) throws IOException{
        return new ServerSocket(port, 50, InetAddress.getByName(host));
    }
    
    public static ServerSocket newServer(String host, int port, int backlog) throws IOException {
        return new ServerSocket(port, backlog, InetAddress.getByName(host));
    }
    public static Socket newSocket(String address, int port) throws IOException {
        return new Socket(InetAddress.getByName(address), port);
    }
    
    //<editor-fold defaultstate="collapsed" desc="HelloWorld-TCP">
    /**
     * create a hellow-world-single-thread TCP server
     * @param host
     * @param port 
     */
    public static void hwServer(String host, int port)
    {
        ServerSocket server=null;
        Socket client=null;
        
        InputStream in=null;
        OutputStream out=null;
        try
        {
            server=TCP.newServer(host, port);
            System.out.println("Server start at "+host+':'+port);
            byte[] inbuf=new byte[1024];
            while(true)
            {
                client=server.accept();
                System.out.println(client.toString());
                in=client.getInputStream();
                int len=in.read(inbuf);
                if(len!=-1)
                {
                    out=client.getOutputStream();
                    String msg="Received:"+new String(inbuf,0,len);
                    System.out.println(msg);
                    out.write(msg.getBytes());
                }
                client.shutdownInput();
                client.shutdownOutput();
                client.close();
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(server!=null) server.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    /**
     * create a hellow-word-multi-thread TCP Server
     * @param host
     * @param port 
     */
    public static void hwMTServer(String host, int port)
    {
        ServerSocket server=null;
        Socket client=null;
        try
        {
            server=TCP.newServer(host, port);
            System.out.println("Server start at "+host+':'+port);
            while(true)
            {
                client=server.accept();
                System.out.println(client.toString());
                TCPService ts=new TCPService(client);
                Thread t=new Thread(ts);
                t.start();
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(server!=null) server.close();
            }
            catch(IOException e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    /**
     * create a TCP client, and send a msg to he specific Server
     * @param address
     * @param port
     * @param msg 
     */
    public static void hwClient(String address, int port, String msg)
    {
        Socket client=null;
        OutputStream out=null;
        InputStream in=null;
        try
        {
            client=new Socket(InetAddress.getByName(address), port);
            
            out=client.getOutputStream();
            out.write(msg.getBytes());
            
            in=client.getInputStream();
            byte[] buf=new byte[1024];
            int len=in.read(buf);
            if(len!=-1)
            {
                String recv=new String(buf,0,len);
                System.out.println(recv);
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
    public static void send(String address, int port, String msg)
    {
        Socket client=null;
        OutputStream out=null;
        try
        {
            client=new Socket(InetAddress.getByName(address), port);
            
            out=client.getOutputStream();
            out.write(msg.getBytes());
        }
        catch(IOException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(out!=null) out.close();
            }
            catch(Exception e) {e.printStackTrace();}
        }
    }
    //</editor-fold>
}
