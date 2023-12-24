(window.webpackJsonp=window.webpackJsonp||[]).push([[93],{"8li0":function(module,e,n){"use strict";n.d(e,"b",(function(){return o}));var t=n("vaU+"),r=n("/WoZ"),a=n("tEn/"),s={isLaunched:function(e){return!!e.launchedAt&&e.launchedAt<=Date.now()},isPreEnrollmentEnabled:function(e){return!this.isLaunched(e)&&"preEnrollmentEnabledAt"in e},isSessionsModeEnabled:function(e){return void 0!==e.sessionsEnabledAt&&e.sessionsEnabledAt<=Date.now()},isVerificationEnabled:function(e){return!!e.verificationEnabledAt&&e.verificationEnabledAt<=Date.now()},isPrivate:function(e){return Object(t.g)(e)},isFullC4CPartner:function(e){var n=e.isC4CPartner,t=e.isPrivateAuthoringPartner;return n&&!t},getCourseCatalogType:function(e){var n=r.a.PUBLIC;return s.isPrivate(e)?n=r.a.PRIVATE:e.isLimitedToEnterprise&&(n=r.a.ENTERPRISE),n},isProject:function(e){return[a.a.PROJECT,a.a.RHYME_PROJECT].includes(e)},isPrivatelyAuthoredCourse:function(e,n){return s.getCourseCatalogType(e)===r.a.ENTERPRISE&&n.typeName===a.a.STANDARD_COURSE&&e.isLimitedToEnterprise&&e.isRestrictedMembership}};e.a=s;s.isLaunched,s.isPreEnrollmentEnabled,s.isSessionsModeEnabled,s.isVerificationEnabled,s.isPrivate,s.isFullC4CPartner;var o=s.getCourseCatalogType;s.isProject,s.isPrivatelyAuthoredCourse},"jq+c":function(module,e,n){"use strict";n.r(e);var t=n("VbXa"),r=n.n(t),a=n("q1tI"),s=n("S+eF"),o=n.n(s),i=n("DnuM"),c=n("sQ/U"),u=n("hs7e"),l=n("fw5G"),d=n.n(l),m=n("TKYU"),h=n("welz"),p=n.n(h),f=n("pWop"),v=p.a.Collection.extend({model:f.a,getEnrolled:function(){return this.filter((function(e){return e.get("courseRole")!==f.a.NOT_ENROLLED}))}}),promises_memberships=function(e){var n=function(e){var n=(new d.a).addQueryParam("q","findByUser").addQueryParam("userId",e);return o()(m.a.get(n.toString()))}(e).then((function(e){return new v(e.elements)}));return n.done(),n},E=n("8li0"),b=Object(i.a)("/api/reports.v1",{type:"rest"}),y="teachVisitedAt",g=10080;function getBannerData(){return function(){var e=c.a.get().id;if(!e)return o.a.reject();return promises_memberships(e).then((function(e){var n=e.find((function(e){return e.hasTeachingRole()}));if(n&&"UNIVERSITY_ADMIN"!==n.get("courseRole")){var t=n.get("courseId");return[n,u.a.fromId(t)]}return o.a.reject()})).spread((function(e,n){return E.a.isLaunched(n)?[e,n]:o.a.reject()}))}().spread((function(e,n){return[e,n,getLearnerCounts(e,n)]}))}function shouldShow(){var e=localStorage[y];if(e){var n=new Date(parseInt(e)),t=new Date(Date.now());return 60*t.getHours()+t.getMinutes()-(60*n.getHours()+n.getMinutes())>g}return!0}function getLearnerCounts(e,n){return o()(b.get("Course~".concat(n.id,"~activity_learner_counts"))).then((function(t){return function(e){return e.elements&&e.elements[0]&&e.elements[0].body&&e.elements[0].body.latest&&e.elements[0].body.latest.starter_ever_count&&e.elements[0].body.latest.active_learner_past_1w_count&&e.elements[0].body.latest.visitor_ever_count&&e.elements[0].body["1w_ago"]&&e.elements[0].body["1w_ago"].starter_ever_count}(t)?t.elements[0].body:o.a.reject({membership:e,course:n})}))}n("wab5");var w,C=n("17x9"),T=n.n(C),L=n("kvW3"),B=function(e){function TotalLearnerBanner(){return e.apply(this,arguments)||this}return r()(TotalLearnerBanner,e),TotalLearnerBanner.prototype.render=function(){var e=this.props.course,n=e.name,t="/teach/".concat(e.slug),r=this.props.learnerCounts.latest.visitor_ever_count;return a.createElement("div",{className:"rc-TotalLearnerBanner"},"A total of"," ",a.createElement("a",{href:t},a.createElement("strong",{className:"c-teach-banner-learner-count"},a.createElement(L.c,{value:r})," learners"))," ","are enrolled in ",a.createElement("span",{className:"c-teach-banner-course-name"},n),". View more on the"," ",a.createElement("a",{href:t},w||(w=a.createElement("strong",null,"Course Dashboard."))))},TotalLearnerBanner}(a.Component);B.propTypes={course:T.a.object.isRequired,learnerCounts:T.a.object.isRequired};var N,A=B,P=function(e){function WeeklyActiveLearnerBanner(){return e.apply(this,arguments)||this}return r()(WeeklyActiveLearnerBanner,e),WeeklyActiveLearnerBanner.prototype.render=function(){var e=this.props.course,n=e.name,t="/teach/".concat(e.slug),r=this.props.learnerCounts.latest.active_learner_past_1w_count;return a.createElement("div",{className:"rc-WeeklyActiveLearnerBanner"},a.createElement("a",{href:t},a.createElement("strong",{className:"c-teach-banner-learner-count"},a.createElement(L.c,{value:r})," learners"))," ","were active in ",a.createElement("span",{className:"c-teach-banner-course-name"},n)," in the past week. View more on the"," ",a.createElement("a",{href:t},N||(N=a.createElement("strong",null,"Course Dashboard."))))},WeeklyActiveLearnerBanner}(a.Component);P.propTypes={course:T.a.object.isRequired,learnerCounts:T.a.object.isRequired};var R,k=P,D=function(e){function WeeklyNewLearnerBanner(){return e.apply(this,arguments)||this}return r()(WeeklyNewLearnerBanner,e),WeeklyNewLearnerBanner.prototype.render=function(){var e,n=this.props.course,t=n.name,r="/teach/".concat(n.slug),s=(e=this.props.learnerCounts).latest.starter_ever_count-e["1w_ago"].starter_ever_count;return a.createElement("div",{className:"rc-WeeklyNewLearnerBanner"},a.createElement("a",{href:r},a.createElement("strong",{className:"c-teach-banner-learner-count"},a.createElement(L.c,{value:s})," learners"))," ","enrolled in ",a.createElement("span",{className:"c-teach-banner-course-name"},t)," in the past week. View more on the"," ",a.createElement("a",{href:r},R||(R=a.createElement("strong",null,"Course Dashboard."))))},WeeklyNewLearnerBanner}(a.Component);D.propTypes={course:T.a.object.isRequired,learnerCounts:T.a.object.isRequired};var j,I,S=[{key:"weeklyActiveLearner",component:k},{key:"weeklyNewLearner",component:D},{key:"totalLearner",component:A}],W=function(e){function TeachBanner(){for(var n,t=arguments.length,r=new Array(t),a=0;a<t;a++)r[a]=arguments[a];return(n=e.call.apply(e,[this].concat(r))||this).state={course:null,membership:null,learnerCounts:{},bannerIndex:-1,dismissed:!1},n.handleDismiss=function(){localStorage[y]=Date.now(),n.setState({dismissed:!0})},n}r()(TeachBanner,e);var n=TeachBanner.prototype;return n.componentDidMount=function(){var e=this;getBannerData().spread((function(n,t,r){var a=Math.round(Math.random()*(S.length-1));e.setState({course:t,membership:n,learnerCounts:r,bannerIndex:a})})).catch((function(){})).done()},n.renderBanner=function(){var e=S[this.state.bannerIndex].component;return(a.createElement(e,{course:this.state.course,learnerCounts:this.state.learnerCounts}))},n.render=function(){return this.state.course&&!this.state.dismissed&&shouldShow()?a.createElement("div",{className:"rc-TeachBanner bt3-alert bt3-alert-info bt3-alert-dismissable"},a.createElement("div",{className:"c-teach-banner-content"},a.createElement("button",{onClick:this.handleDismiss,className:"bt3-close","data-dismiss":"alert","aria-label":"Close"},I||(I=a.createElement("span",{"aria-hidden":"true"},"×"))),this.renderBanner())):j||(j=a.createElement("div",null))},TeachBanner}(a.Component);e.default=W},wab5:function(module,exports,e){e("xTv4")},xTv4:function(module,exports,e){}}]);
//# sourceMappingURL=93.16b057dc9247420fcc86.js.map