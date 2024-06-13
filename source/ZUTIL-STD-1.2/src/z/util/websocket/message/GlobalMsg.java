/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.websocket.message;

/**
 *
 * @author dell
 */
public class GlobalMsg extends Message
{
    //functions-----------------------------------------------------------------
    public GlobalMsg()
    {
        super(HEAD_GLOBAL);
    }
    public GlobalMsg(String srcPartitionId, String srcSessionId, String data) 
    {
        super(HEAD_GLOBAL, srcPartitionId, srcSessionId, data);
    }
}
