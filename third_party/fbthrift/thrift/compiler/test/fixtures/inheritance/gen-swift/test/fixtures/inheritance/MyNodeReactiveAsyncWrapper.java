/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */

package test.fixtures.inheritance;

import java.util.*;

public class MyNodeReactiveAsyncWrapper  extends test.fixtures.inheritance.MyRootReactiveAsyncWrapper
  implements MyNode.Async {
  private MyNode.Reactive _delegate;

  public MyNodeReactiveAsyncWrapper(MyNode.Reactive _delegate) {
    super(_delegate);
    this._delegate = _delegate;
  }

  public MyNodeReactiveAsyncWrapper(org.apache.thrift.ProtocolId _protocolId, reactor.core.publisher.Mono<? extends com.facebook.swift.transport.client.RpcClient> _rpcClient, Map<String, String> _headers, Map<String, String> _persistentHeaders) {
    this(new MyNodeReactiveClient(_protocolId, _rpcClient, _headers, _persistentHeaders));
  }

  @java.lang.Override
  public void close() {
    _delegate.dispose();
  }

  @java.lang.Override
  public com.google.common.util.concurrent.ListenableFuture<Void> doMid() {
      return com.facebook.swift.transport.util.FutureUtil.toListenableFuture(_delegate.doMid());
  }

  @java.lang.Override
  public com.google.common.util.concurrent.ListenableFuture<Void> doMid(
    com.facebook.swift.transport.client.RpcOptions rpcOptions) {
      return com.facebook.swift.transport.util.FutureUtil.toListenableFuture(_delegate.doMid( rpcOptions));
  }

  @java.lang.Override
  public com.google.common.util.concurrent.ListenableFuture<com.facebook.swift.transport.client.ResponseWrapper<Void>> doMidWrapper(
    com.facebook.swift.transport.client.RpcOptions rpcOptions) {
    return com.facebook.swift.transport.util.FutureUtil.toListenableFuture(_delegate.doMidWrapper( rpcOptions));
  }

}
