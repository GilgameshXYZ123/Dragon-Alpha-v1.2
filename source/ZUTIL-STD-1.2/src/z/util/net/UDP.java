/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.net;

import java.io.IOException;
import java.net.DatagramSocket;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

/**
 *
 * @author dell
 */
public class UDP 
{
    //<editor-fold defaultstate="collapsed" desc="class UDPService">
    public static class UDPService implements Runnable
    {
        //columns-------------------------------------------------------------------
        protected DatagramPacket pack;
        
        //constructor---------------------------------------------------------------
        public void setPack(DatagramPacket pack) 
        {
            if(pack==null) throw new NullPointerException();
            this.pack=pack;
        }
        @Override
        public void run() 
        {
            System.out.println(UDP.toString(pack));
        }
    }
    //</editor-fold>
    public static String toString(DatagramPacket dp)
    {
        if(dp==null) return "null";
        StringBuilder sb=new StringBuilder();
        sb.append("UDP-Packet{");
        sb.append("\n\tSocketAddress = ").append(dp.getSocketAddress());
        sb.append("\n\tPort = ").append(dp.getPort());
        sb.append("\n\tLength = ").append(dp.getLength());
        sb.append("\n\tOffset = ").append(dp.getOffset());
        sb.append("\n\tData = ").append(new String(dp.getData()));
        sb.append("\n}");
        return sb.toString();
    }
    public static DatagramPacket newPacket(String msg, String host, int port) throws UnknownHostException
    {
        InetAddress address=InetAddress.getByName(host);
        byte[] buf=msg.getBytes();
        if(buf.length<64)
        {
            byte[] nbuf=new byte[64];
            System.arraycopy(buf, 0, nbuf, 0, buf.length);
            buf=nbuf;
        }
        return new DatagramPacket(buf, buf.length, address, port);
    }
    public static DatagramPacket newPacket(int bsize)
    {
        return new DatagramPacket(new byte[bsize],bsize);
    }
    public static <T extends DatagramSocket> T newSocket(String host, int port) throws IOException
    {
        InetAddress address=InetAddress.getByName(host);
        return (T) new DatagramSocket(port,address);
    }
    //<editor-fold defaultstate="defaultstate" desc="quick-quickSend">
    private static DatagramSocket SENDER=null;
    public static synchronized void initQuickSender()
    {
        try
        {
            if(SENDER==null) SENDER=new DatagramSocket();
        }
        catch(SocketException e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
    public static void quickSend(String msg, String host, int port) throws IOException
    {
        DatagramPacket pack=UDP.newPacket(msg, host, port);
        SENDER.send(pack);
    }
    public static void quickAck(String keyWord, DatagramPacket pack) throws IOException
    {
        pack.setData(keyWord.getBytes());
        SENDER.send(pack);
    }
    public static boolean checkAck(String keyWord, DatagramPacket pack) throws IOException
    {
        String msg=new String(pack.getData(),0,pack.getLength());
        return keyWord.equals(msg);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Text-HellowWorld-UDP">
    /**
     * create a hellow-word UDP Receiver
     * @param host
     * @param port 
     */
    public static void hwReceiver(String host, int port)
    {
        DatagramSocket server=null;
        try
        {
            server=UDP.newSocket(host, port);
            DatagramPacket pack=null;
            System.out.println("HellowWorld-Server Start at "+host+':'+port);
            while(true)
            {
                pack=UDP.newPacket(1024);
                server.receive(pack);
                System.out.println(UDP.toString(pack));
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
                System.out.println("Hellow-World Server closed at "+host+':'+port);
            }
            catch(Exception e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    /**
     * create a hellow-world UDP Sende, and quickSend all elements in the 
     * input Array {@coe msg} to the specific UDP receiver
     * @param host
     * @param port
     * @param msg 
     */
    public static void hwSender(String host, int port,String[] msg)
    {
        DatagramSocket client=null;
        DatagramPacket pack=null;
        try
        {
            client=new DatagramSocket();
            for(int i=0;i<msg.length;i++)
            {
                pack=UDP.newPacket(msg[i], host, port);
                client.send(pack);
                System.out.println("SEND_DATA:"+msg[i]);
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
                if(client!=null) client.close();
            }
            catch(Exception e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    /**
     * create a hellow-word-Mutli-Thread UDP Receiver
     * @param host
     * @param port
     * @param us 
     */
    public static void hwMTServer(String host, int port, UDPService us) 
    {
        DatagramSocket server=null;
        try
        {
            server=UDP.newSocket(host, port);
            DatagramPacket pack=null;
            System.out.println("HellowWorld-Server Start at "+host+':'+port);
            while(true)
            {
                pack=UDP.newPacket(1024);
                server.receive(pack);
                us.setPack(pack);
                Thread t=new Thread(us);
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
                System.out.println("Hellow-World Server closed at "+host+':'+port);
            }
            catch(Exception e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    /**
     * create a hellow-word-Mutli-Thread UDP Receiver
     * @param host
     * @param port
     */
    public static void hwMTServer(String host, int port) 
    {
        UDP.hwMTServer(host, port, new UDPService());
    }
    /**
     * create a hellow-word-Multi-Thread UDP Server, and it will resend
     * the received datapacket.
     * @param host
     * @param port 
     */
    public static void hwMTAckServer(String host, int port)
    {
        UDPService us=new UDPService()
        {
            @Override
            public void run() 
            {
                try
                {
                    System.out.println("\nReceived:"+UDP.toString(pack));
                    Thread t=new Thread(new Runnable() 
                    {
                        @Override
                        public void run() 
                        {
                            DatagramSocket acker=null;
                            try
                            {
                                acker=new DatagramSocket();
                                //send msg--------------------------------------
                                pack.setData("Need Your ACK".getBytes());
                                acker.send(pack);
                                
                                //wait for quickAck----------------------------------
                                acker.receive(pack);
                                String data=new String(pack.getData(),0,pack.getLength());
                                System.out.println("ACK Data:"+data);
                                if("ACK".equals(data))
                                    System.out.println("\nACK:"+UDP.toString(pack));
                                else System.out.println("\nWrong ACK!");
                            }
                            catch(IOException e)
                            {
                                System.err.println(e);
                                System.out.println("failed");
                            }
                            finally
                            {
                                if(acker!=null) acker.close();
                            }
                        }
                    });
                    t.start();
                    System.out.println("Wait for ACK:");
                    Thread.sleep(1000);
                    if(t.isAlive())
                    {
                        System.out.println("Interrupted");
                         t.interrupt();
                   }
                }
                catch(InterruptedException e)
                {
                    e.printStackTrace();
                }
            }
        };
        UDP.hwMTServer(host, port, us);
    }
      /**
     * create a hellow-word-Single-Thread UDP Server, and it will resend
     * the received datapacket.
     * @param host
     * @param port 
     */
    public static void hwACKClient(String host, int port,String[] msg)
    {
        DatagramSocket client=null;
        DatagramPacket pack=null;
        try
        {
            client=new DatagramSocket();
            for(int i=0;i<msg.length;i++)
            {
                pack=UDP.newPacket(msg[i], host, port);
                client.send(pack);
                System.out.println("SEND_DATA:"+msg[i]);
                
                //recv msg from hwMTACK server----------------------------------
                client.receive(pack);
                System.out.println("Received:"+UDP.toString(pack));
                System.out.println("Send ACK.");
                byte[] buf="ACK".getBytes();
                pack.setData(buf);
                pack.setLength(buf.length);
                client.send(pack);
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
                if(client!=null) client.close();
            }
            catch(Exception e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    
    //</editor-fold>
}
