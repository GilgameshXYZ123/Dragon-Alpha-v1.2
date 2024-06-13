/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.websocket.message;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.util.HashMap;
/**
 *
 * @author dell
 */
public abstract class Message implements Serializable
{
    //MessageParseException-----------------------------------------------------
    public static class MessageParseException extends RuntimeException
    {
        public static final String MSG="MessageParseException";
        
        public MessageParseException() 
        {
            super();
        }
        public MessageParseException(String msg)
        {
            super(MSG+msg);
        }
    }
    
    //static--------------------------------------------------------------------
    public static final String HEAD_GLOBAL="global";
    public static final String HEAD_LOCAL="local";
    public static final String HEAD_P2P="p2p";
    
    protected static final HashMap<String, Constructor> header=new HashMap<>();
    //<editor-fold defaultstate="collapsed" desc="Init-Code">
    static
    {
        synchronized(Message.class)
        {
            try
            {
                Class[] clazz={P2PMsg.class,LocalMsg.class,GlobalMsg.class};
                String[] key={HEAD_P2P,HEAD_LOCAL,HEAD_GLOBAL};
                for(int i=0;i<clazz.length;i++)
                {
                    header.put(key[i], clazz[i].getConstructor());
                }
                System.out.println(Message.showMsgHeader());
            }
            catch(NoSuchMethodException | SecurityException e)
            {
                e.printStackTrace();
                throw new RuntimeException(e);
            }
        }
    }
    //</editor-fold>
    
    public static String showMsgHeader()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("MSGHeader = {\n");
        header.forEach((String key, Constructor val)->
            {sb.append(key).append(" = ").append(val).append("\n");});
        sb.append("}");
        return sb.toString();
    }
    
    //columns-------------------------------------------------------------------
    protected String head;
    String srcPartitionId;
    String srcSessionId;
    protected String data;
    
    //constructors--------------------------------------------------------------
    protected Message(String head) 
    {
        this.head=head;
    }
    public Message(String head,String srcPartitionId, String srcSessionId, String data) 
    {
        this.head=head;
        this.srcPartitionId = srcPartitionId;
        this.srcSessionId = srcSessionId;
        this.data = data;
    }
    //getter--------------------------------------------------------------------
    public String getHead() 
    {
        return head;
    }
    public String getData() 
    {
        return data;
    }
    public String getSrcPartitionId() 
    {
        return srcPartitionId;
    }
    public String getSrcSessionId() 
    {
        return srcSessionId;
    }
    //String--------------------------------------------------------------------
    @Override
    public String toString()
    {
        return head+':'+this.srcPartitionId+':'+this.srcSessionId+':'+this.data;
    }
    public void loadData(String[] args)
    {
        this.srcPartitionId=args[1];
        this.srcSessionId=args[2];
        this.data=args[3];
    }
    //static parser-------------------------------------------------------------
    public static <T extends Message> T valueOf(String line) throws Exception
    {
        String tk[]=line.split(":");
        Constructor con=header.get(tk[0]);
        if(con==null) throw new MessageParseException("Unknown Header:"+tk[0]);
        T val=(T) con.newInstance();
        val.loadData(tk);
        return val;
    }
}
