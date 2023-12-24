/*
 * Injected into third party widgets to allow communication with Coursera.
 * Talks to coursera-connect-parent.js
 */

// eslint-disable-next-line
var courseraApi = (function() {

  var _messageToken;
  var _parentOrigin;
  var _isInitialized = false;
  var _isConnectedToParent = false;

  /**
   * Generates a random 10-character id.
   */
  function _generateId() {
    return Array(11)
      .join((Math.random().toString(36) + '00000000000000000').slice(2, 18))
      .slice(0, 10);
  }

  /**
   * Send an authenticated message to the parent frame.
   */
  function _sendMessageToParent(requestType, data) {
    if (!_isConnectedToParent || !_messageToken) {
      return '';
    }

    if (window && window.parent) {
      var requestId = _generateId();
      var messageData = {
        token: _messageToken,
        id: requestId,
        type: requestType,
        body: data,
      };
      window.parent.postMessage(messageData, _parentOrigin);

      return requestId;
    }

    return '';
  }

  /**
   * Listens for authentication from parent and sets the `_parentOrigin` and `_messageToken`
   * variables, which authenticate messages from the child frame.
   */
  function _listenForParentInit() {
    var onParentMessage = function(event) {
      var parentResponse = event.data;
      if (!parentResponse || !parentResponse.token || !parentResponse.id) {
        return;
      }

      switch (parentResponse.type) {
        case 'INIT_CHILD': {
          if (!_isInitialized) {
            _messageToken = parentResponse.token;
            _parentOrigin = event.origin;
            _isInitialized = true;

            window.parent.postMessage(parentResponse, _parentOrigin);
          } else {
            // This is an event meant for another plugin. This can happen when there are multiple
            // plugins on a single page (such as quizzes), and usually when a second plugin is
            // receiving an INIT_CHILD message before this plugin receives its INIT_COMPLETE
            // message. Before adding _isInitialized, this would overwrite the first plugin's
            // _messageToken, meaning it would never accept anymore messages because everything,
            // most notably INIT_COMPLETE, checks for `parentResponse.token === _messageToken`.
          }
          break;
        }
        case 'INIT_COMPLETE': {
          if (parentResponse.token === _messageToken) {
            _isConnectedToParent = true;
            window.removeEventListener('message', onParentMessage);
          }
          break;
        }
        case 'ERROR': {
          window.removeEventListener('message', onParentMessage);
          break;
        }
        default: {
          break;
        }
      }
    };

    window.addEventListener('message', onParentMessage);
  }


  /**
   * Listen for a response with the specified `requestId` from the parent.
   * Pass the message to `callback` on receipt and de-register the listener.
   */
  function _listenForParentResponse(messageId, onSuccess, onError) {
    var onParentReply = function(event) {
      var parentResponse = event.data;
      if (!parentResponse) {
        onError('NO_DATA_RECEIVED');
        return;
      }

      if (parentResponse.token === _messageToken && parentResponse.id === messageId) {
        if (parentResponse.type === 'ERROR') {
          if (parentResponse.body && parentResponse.body.errorCode) {
            onError(parentResponse.body.errorCode);
          }
        } else {
          onSuccess(parentResponse.body);
        }
        window.removeEventListener('message', onParentReply);
      }
    };

    window.addEventListener('message', onParentReply);
  }

  // Public methods

  // TODO(Holly): Figure out standard for documenting Javascript functions (FLEX-10664)

  /**
   * Call allowed parent methods.
   * options {
   *  type string
   *  [onSuccess] (allowedMethods: Array<string>)
   *  [onError] (errorMessage: string)
   *  [data] any
   * }
   */
  function callMethod(options) {
    // eslint-disable-next-line prefer-destructuring
    var type = options.type;

    var onSuccess = options.onSuccess || function() {};
    var onError = options.onError || function() {};
    var data = options.data || {};

    if (!type) {
      onError('MESSAGE_TYPE_NOT_DEFINED');
    } else if (!_isConnectedToParent) {
      // Try again later, when initialization is hopefully done.
      setTimeout(function() {
        callMethod(options);
      }, 500);
    } else {
      var messageId = _sendMessageToParent(type, data);
      _listenForParentResponse(messageId, onSuccess, onError);
    }
  }

  // Establish communication with the container automatically.
  _listenForParentInit();

  return {
    callMethod: callMethod,
  };
})();
