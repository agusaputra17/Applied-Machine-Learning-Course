��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.12v2.15.0-11-g63f5a65c7cd8��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
.AdamW/v/recommender_net/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/v/recommender_net/embedding_3/embeddings/*
dtype0*
shape:	�W*?
shared_name0.AdamW/v/recommender_net/embedding_3/embeddings
�
BAdamW/v/recommender_net/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/v/recommender_net/embedding_3/embeddings*
_output_shapes
:	�W*
dtype0
�
.AdamW/m/recommender_net/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/m/recommender_net/embedding_3/embeddings/*
dtype0*
shape:	�W*?
shared_name0.AdamW/m/recommender_net/embedding_3/embeddings
�
BAdamW/m/recommender_net/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/m/recommender_net/embedding_3/embeddings*
_output_shapes
:	�W*
dtype0
�
.AdamW/v/recommender_net/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/v/recommender_net/embedding_2/embeddings/*
dtype0*
shape:	�W*?
shared_name0.AdamW/v/recommender_net/embedding_2/embeddings
�
BAdamW/v/recommender_net/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/v/recommender_net/embedding_2/embeddings*
_output_shapes
:	�W*
dtype0
�
.AdamW/m/recommender_net/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/m/recommender_net/embedding_2/embeddings/*
dtype0*
shape:	�W*?
shared_name0.AdamW/m/recommender_net/embedding_2/embeddings
�
BAdamW/m/recommender_net/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/m/recommender_net/embedding_2/embeddings*
_output_shapes
:	�W*
dtype0
�
.AdamW/v/recommender_net/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/v/recommender_net/embedding_1/embeddings/*
dtype0*
shape:
��*?
shared_name0.AdamW/v/recommender_net/embedding_1/embeddings
�
BAdamW/v/recommender_net/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/v/recommender_net/embedding_1/embeddings* 
_output_shapes
:
��*
dtype0
�
.AdamW/m/recommender_net/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *?

debug_name1/AdamW/m/recommender_net/embedding_1/embeddings/*
dtype0*
shape:
��*?
shared_name0.AdamW/m/recommender_net/embedding_1/embeddings
�
BAdamW/m/recommender_net/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp.AdamW/m/recommender_net/embedding_1/embeddings* 
_output_shapes
:
��*
dtype0
�
,AdamW/v/recommender_net/embedding/embeddingsVarHandleOp*
_output_shapes
: *=

debug_name/-AdamW/v/recommender_net/embedding/embeddings/*
dtype0*
shape:
��*=
shared_name.,AdamW/v/recommender_net/embedding/embeddings
�
@AdamW/v/recommender_net/embedding/embeddings/Read/ReadVariableOpReadVariableOp,AdamW/v/recommender_net/embedding/embeddings* 
_output_shapes
:
��*
dtype0
�
,AdamW/m/recommender_net/embedding/embeddingsVarHandleOp*
_output_shapes
: *=

debug_name/-AdamW/m/recommender_net/embedding/embeddings/*
dtype0*
shape:
��*=
shared_name.,AdamW/m/recommender_net/embedding/embeddings
�
@AdamW/m/recommender_net/embedding/embeddings/Read/ReadVariableOpReadVariableOp,AdamW/m/recommender_net/embedding/embeddings* 
_output_shapes
:
��*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
&recommender_net/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *7

debug_name)'recommender_net/embedding_3/embeddings/*
dtype0*
shape:	�W*7
shared_name(&recommender_net/embedding_3/embeddings
�
:recommender_net/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_3/embeddings*
_output_shapes
:	�W*
dtype0
�
&recommender_net/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *7

debug_name)'recommender_net/embedding_2/embeddings/*
dtype0*
shape:	�W*7
shared_name(&recommender_net/embedding_2/embeddings
�
:recommender_net/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_2/embeddings*
_output_shapes
:	�W*
dtype0
�
&recommender_net/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *7

debug_name)'recommender_net/embedding_1/embeddings/*
dtype0*
shape:
��*7
shared_name(&recommender_net/embedding_1/embeddings
�
:recommender_net/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_1/embeddings* 
_output_shapes
:
��*
dtype0
�
$recommender_net/embedding/embeddingsVarHandleOp*
_output_shapes
: *5

debug_name'%recommender_net/embedding/embeddings/*
dtype0*
shape:
��*5
shared_name&$recommender_net/embedding/embeddings
�
8recommender_net/embedding/embeddings/Read/ReadVariableOpReadVariableOp$recommender_net/embedding/embeddings* 
_output_shapes
:
��*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$recommender_net/embedding/embeddings&recommender_net/embedding_1/embeddings&recommender_net/embedding_2/embeddings&recommender_net/embedding_3/embeddings*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_368188

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
		user_bias

resto_embedding

resto_bias
	optimizer

signatures*
 
0
1
2
3*
 
0
1
2
3*

0
1* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

embeddings*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

embeddings*
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

embeddings*
�
3
_variables
4_iterations
5_learning_rate
6_index_dict
7
_momentums
8_velocities
9_update_step_xla*

:serving_default* 
d^
VARIABLE_VALUE$recommender_net/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_2/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_3/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

;trace_0* 

<trace_0* 
* 
 
0
	1

2
3*

=0
>1*
* 
* 
* 
* 

0*

0*
	
0* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 

0*

0*
* 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 

0*

0*
	
0* 
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 

0*

0*
* 
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
C
40
[1
\2
]3
^4
_5
`6
a7
b8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
[0
]1
_2
a3*
 
\0
^1
`2
b3*
* 
* 
* 
* 
8
c	variables
d	keras_api
	etotal
	fcount*
8
g	variables
h	keras_api
	itotal
	jcount*
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
wq
VARIABLE_VALUE,AdamW/m/recommender_net/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,AdamW/v/recommender_net/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/m/recommender_net/embedding_1/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/v/recommender_net/embedding_1/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/m/recommender_net/embedding_2/embeddings1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/v/recommender_net/embedding_2/embeddings1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/m/recommender_net/embedding_3/embeddings1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE.AdamW/v/recommender_net/embedding_3/embeddings1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

c	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

g	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$recommender_net/embedding/embeddings&recommender_net/embedding_1/embeddings&recommender_net/embedding_2/embeddings&recommender_net/embedding_3/embeddings	iterationlearning_rate,AdamW/m/recommender_net/embedding/embeddings,AdamW/v/recommender_net/embedding/embeddings.AdamW/m/recommender_net/embedding_1/embeddings.AdamW/v/recommender_net/embedding_1/embeddings.AdamW/m/recommender_net/embedding_2/embeddings.AdamW/v/recommender_net/embedding_2/embeddings.AdamW/m/recommender_net/embedding_3/embeddings.AdamW/v/recommender_net/embedding_3/embeddingstotal_1count_1totalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_368410
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$recommender_net/embedding/embeddings&recommender_net/embedding_1/embeddings&recommender_net/embedding_2/embeddings&recommender_net/embedding_3/embeddings	iterationlearning_rate,AdamW/m/recommender_net/embedding/embeddings,AdamW/v/recommender_net/embedding/embeddings.AdamW/m/recommender_net/embedding_1/embeddings.AdamW/v/recommender_net/embedding_1/embeddings.AdamW/m/recommender_net/embedding_2/embeddings.AdamW/v/recommender_net/embedding_2/embeddings.AdamW/m/recommender_net/embedding_3/embeddings.AdamW/v/recommender_net/embedding_3/embeddingstotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_368473��
�
�
G__inference_embedding_2_layer_call_and_return_conditional_losses_368265

inputs	*
embedding_lookup_368256:	�W
identity��embedding_lookup�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_368256inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368256*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_368256*
_output_shapes
:	�W*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^embedding_lookupI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:&"
 
_user_specified_name368256:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_embedding_3_layer_call_fn_368272

inputs	
unknown:	�W
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_368071o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368268:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_embedding_1_layer_call_and_return_conditional_losses_368037

inputs	+
embedding_lookup_368032:
��
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_368032inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368032*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name368032:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�a
�
!__inference__wrapped_model_368004
input_1	E
1recommender_net_embedding_embedding_lookup_367936:
��G
3recommender_net_embedding_1_embedding_lookup_367944:
��F
3recommender_net_embedding_2_embedding_lookup_367952:	�WF
3recommender_net_embedding_3_embedding_lookup_367960:	�W
identity��*recommender_net/embedding/embedding_lookup�,recommender_net/embedding_1/embedding_lookup�,recommender_net/embedding_2/embedding_lookup�,recommender_net/embedding_3/embedding_lookupt
#recommender_net/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%recommender_net/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%recommender_net/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_sliceStridedSliceinput_1,recommender_net/strided_slice/stack:output:0.recommender_net/strided_slice/stack_1:output:0.recommender_net/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
*recommender_net/embedding/embedding_lookupResourceGather1recommender_net_embedding_embedding_lookup_367936&recommender_net/strided_slice:output:0*
Tindices0	*D
_class:
86loc:@recommender_net/embedding/embedding_lookup/367936*'
_output_shapes
:���������*
dtype0�
3recommender_net/embedding/embedding_lookup/IdentityIdentity3recommender_net/embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:���������v
%recommender_net/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'recommender_net/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_1StridedSliceinput_1.recommender_net/strided_slice_1/stack:output:00recommender_net/strided_slice_1/stack_1:output:00recommender_net/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_1/embedding_lookupResourceGather3recommender_net_embedding_1_embedding_lookup_367944(recommender_net/strided_slice_1:output:0*
Tindices0	*F
_class<
:8loc:@recommender_net/embedding_1/embedding_lookup/367944*'
_output_shapes
:���������*
dtype0�
5recommender_net/embedding_1/embedding_lookup/IdentityIdentity5recommender_net/embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:���������v
%recommender_net/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_2StridedSliceinput_1.recommender_net/strided_slice_2/stack:output:00recommender_net/strided_slice_2/stack_1:output:00recommender_net/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_2/embedding_lookupResourceGather3recommender_net_embedding_2_embedding_lookup_367952(recommender_net/strided_slice_2:output:0*
Tindices0	*F
_class<
:8loc:@recommender_net/embedding_2/embedding_lookup/367952*'
_output_shapes
:���������*
dtype0�
5recommender_net/embedding_2/embedding_lookup/IdentityIdentity5recommender_net/embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:���������v
%recommender_net/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_3StridedSliceinput_1.recommender_net/strided_slice_3/stack:output:00recommender_net/strided_slice_3/stack_1:output:00recommender_net/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_3/embedding_lookupResourceGather3recommender_net_embedding_3_embedding_lookup_367960(recommender_net/strided_slice_3:output:0*
Tindices0	*F
_class<
:8loc:@recommender_net/embedding_3/embedding_lookup/367960*'
_output_shapes
:���������*
dtype0�
5recommender_net/embedding_3/embedding_lookup/IdentityIdentity5recommender_net/embedding_3/embedding_lookup:output:0*
T0*'
_output_shapes
:���������o
recommender_net/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       a
recommender_net/Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB �
recommender_net/Tensordot/ShapeShape<recommender_net/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::��i
'recommender_net/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/GatherV2GatherV2(recommender_net/Tensordot/Shape:output:0'recommender_net/Tensordot/free:output:00recommender_net/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: k
)recommender_net/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_1GatherV2(recommender_net/Tensordot/Shape:output:0'recommender_net/Tensordot/axes:output:02recommender_net/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
recommender_net/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
recommender_net/Tensordot/ProdProd+recommender_net/Tensordot/GatherV2:output:0(recommender_net/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!recommender_net/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_1Prod-recommender_net/Tensordot/GatherV2_1:output:0*recommender_net/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%recommender_net/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 recommender_net/Tensordot/concatConcatV2'recommender_net/Tensordot/free:output:0'recommender_net/Tensordot/axes:output:0.recommender_net/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
recommender_net/Tensordot/stackPack'recommender_net/Tensordot/Prod:output:0)recommender_net/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
#recommender_net/Tensordot/transpose	Transpose<recommender_net/embedding/embedding_lookup/Identity:output:0)recommender_net/Tensordot/concat:output:0*
T0*'
_output_shapes
:����������
!recommender_net/Tensordot/ReshapeReshape'recommender_net/Tensordot/transpose:y:0(recommender_net/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������q
 recommender_net/Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       c
 recommender_net/Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB �
!recommender_net/Tensordot/Shape_1Shape>recommender_net/embedding_2/embedding_lookup/Identity:output:0*
T0*
_output_shapes
::��k
)recommender_net/Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_2GatherV2*recommender_net/Tensordot/Shape_1:output:0)recommender_net/Tensordot/free_1:output:02recommender_net/Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: k
)recommender_net/Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_3GatherV2*recommender_net/Tensordot/Shape_1:output:0)recommender_net/Tensordot/axes_1:output:02recommender_net/Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!recommender_net/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_2Prod-recommender_net/Tensordot/GatherV2_2:output:0*recommender_net/Tensordot/Const_2:output:0*
T0*
_output_shapes
: k
!recommender_net/Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_3Prod-recommender_net/Tensordot/GatherV2_3:output:0*recommender_net/Tensordot/Const_3:output:0*
T0*
_output_shapes
: i
'recommender_net/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/concat_1ConcatV2)recommender_net/Tensordot/axes_1:output:0)recommender_net/Tensordot/free_1:output:00recommender_net/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
!recommender_net/Tensordot/stack_1Pack)recommender_net/Tensordot/Prod_3:output:0)recommender_net/Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
%recommender_net/Tensordot/transpose_1	Transpose>recommender_net/embedding_2/embedding_lookup/Identity:output:0+recommender_net/Tensordot/concat_1:output:0*
T0*'
_output_shapes
:����������
#recommender_net/Tensordot/Reshape_1Reshape)recommender_net/Tensordot/transpose_1:y:0*recommender_net/Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
 recommender_net/Tensordot/MatMulMatMul*recommender_net/Tensordot/Reshape:output:0,recommender_net/Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������i
'recommender_net/Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/concat_2ConcatV2+recommender_net/Tensordot/GatherV2:output:0-recommender_net/Tensordot/GatherV2_2:output:00recommender_net/Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: �
recommender_net/TensordotReshape*recommender_net/Tensordot/MatMul:product:0+recommender_net/Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
recommender_net/addAddV2"recommender_net/Tensordot:output:0>recommender_net/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:����������
recommender_net/add_1AddV2recommender_net/add:z:0>recommender_net/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������o
recommender_net/SigmoidSigmoidrecommender_net/add_1:z:0*
T0*'
_output_shapes
:���������j
IdentityIdentityrecommender_net/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^recommender_net/embedding/embedding_lookup-^recommender_net/embedding_1/embedding_lookup-^recommender_net/embedding_2/embedding_lookup-^recommender_net/embedding_3/embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2X
*recommender_net/embedding/embedding_lookup*recommender_net/embedding/embedding_lookup2\
,recommender_net/embedding_1/embedding_lookup,recommender_net/embedding_1/embedding_lookup2\
,recommender_net/embedding_2/embedding_lookup,recommender_net/embedding_2/embedding_lookup2\
,recommender_net/embedding_3/embedding_lookup,recommender_net/embedding_3/embedding_lookup:&"
 
_user_specified_name367960:&"
 
_user_specified_name367952:&"
 
_user_specified_name367944:&"
 
_user_specified_name367936:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�]
�
K__inference_recommender_net_layer_call_and_return_conditional_losses_368123
input_1	$
embedding_368023:
��&
embedding_1_368038:
��%
embedding_2_368057:	�W%
embedding_3_368072:	�W
identity��!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_368023*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_368022f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_368038*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_368037f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSliceinput_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_368057*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_368056f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_3StridedSliceinput_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_368072*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_368071_
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       Q
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB w
Tensordot/ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	Transpose*embedding/StatefulPartitionedCall:output:0Tensordot/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������a
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB {
Tensordot/Shape_1Shape,embedding_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��[
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1:output:0Tensordot/free_1:output:0"Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1:output:0Tensordot/axes_1:output:0"Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_2ProdTensordot/GatherV2_2:output:0Tensordot/Const_2:output:0*
T0*
_output_shapes
: [
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_3ProdTensordot/GatherV2_3:output:0Tensordot/Const_3:output:0*
T0*
_output_shapes
: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/axes_1:output:0Tensordot/free_1:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
Tensordot/stack_1PackTensordot/Prod_3:output:0Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose_1	Transpose,embedding_2/StatefulPartitionedCall:output:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:����������
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������Y
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_2ConcatV2Tensordot/GatherV2:output:0Tensordot/GatherV2_2:output:0 Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: n
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
addAddV2Tensordot:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������w
add_1AddV2add:z:0,embedding_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_368023* 
_output_shapes
:
��*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_2_368057*
_output_shapes
:	�W*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:&"
 
_user_specified_name368072:&"
 
_user_specified_name368057:&"
 
_user_specified_name368038:&"
 
_user_specified_name368023:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
$__inference_signature_wrapper_368188
input_1	
unknown:
��
	unknown_0:
��
	unknown_1:	�W
	unknown_2:	�W
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_368004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368184:&"
 
_user_specified_name368182:&"
 
_user_specified_name368180:&"
 
_user_specified_name368178:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
,__inference_embedding_2_layer_call_fn_368253

inputs	
unknown:	�W
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_2_layer_call_and_return_conditional_losses_368056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368249:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
0__inference_recommender_net_layer_call_fn_368136
input_1	
unknown:
��
	unknown_0:
��
	unknown_1:	�W
	unknown_2:	�W
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_recommender_net_layer_call_and_return_conditional_losses_368123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368132:&"
 
_user_specified_name368130:&"
 
_user_specified_name368128:&"
 
_user_specified_name368126:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
G__inference_embedding_3_layer_call_and_return_conditional_losses_368071

inputs	*
embedding_lookup_368066:	�W
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_368066inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368066*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name368066:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_embedding_layer_call_and_return_conditional_losses_368022

inputs	+
embedding_lookup_368013:
��
identity��embedding_lookup�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_368013inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368013*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_368013* 
_output_shapes
:
��*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������~
NoOpNoOp^embedding_lookupG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:&"
 
_user_specified_name368013:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�\
�
"__inference__traced_restore_368473
file_prefixI
5assignvariableop_recommender_net_embedding_embeddings:
��M
9assignvariableop_1_recommender_net_embedding_1_embeddings:
��L
9assignvariableop_2_recommender_net_embedding_2_embeddings:	�WL
9assignvariableop_3_recommender_net_embedding_3_embeddings:	�W&
assignvariableop_4_iteration:	 *
 assignvariableop_5_learning_rate: S
?assignvariableop_6_adamw_m_recommender_net_embedding_embeddings:
��S
?assignvariableop_7_adamw_v_recommender_net_embedding_embeddings:
��U
Aassignvariableop_8_adamw_m_recommender_net_embedding_1_embeddings:
��U
Aassignvariableop_9_adamw_v_recommender_net_embedding_1_embeddings:
��U
Bassignvariableop_10_adamw_m_recommender_net_embedding_2_embeddings:	�WU
Bassignvariableop_11_adamw_v_recommender_net_embedding_2_embeddings:	�WU
Bassignvariableop_12_adamw_m_recommender_net_embedding_3_embeddings:	�WU
Bassignvariableop_13_adamw_v_recommender_net_embedding_3_embeddings:	�W%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp5assignvariableop_recommender_net_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp9assignvariableop_1_recommender_net_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp9assignvariableop_2_recommender_net_embedding_2_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_recommender_net_embedding_3_embeddingsIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_iterationIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_learning_rateIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp?assignvariableop_6_adamw_m_recommender_net_embedding_embeddingsIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp?assignvariableop_7_adamw_v_recommender_net_embedding_embeddingsIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpAassignvariableop_8_adamw_m_recommender_net_embedding_1_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpAassignvariableop_9_adamw_v_recommender_net_embedding_1_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpBassignvariableop_10_adamw_m_recommender_net_embedding_2_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpBassignvariableop_11_adamw_v_recommender_net_embedding_2_embeddingsIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpBassignvariableop_12_adamw_m_recommender_net_embedding_3_embeddingsIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpBassignvariableop_13_adamw_v_recommender_net_embedding_3_embeddingsIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:NJ
H
_user_specified_name0.AdamW/v/recommender_net/embedding_3/embeddings:NJ
H
_user_specified_name0.AdamW/m/recommender_net/embedding_3/embeddings:NJ
H
_user_specified_name0.AdamW/v/recommender_net/embedding_2/embeddings:NJ
H
_user_specified_name0.AdamW/m/recommender_net/embedding_2/embeddings:N
J
H
_user_specified_name0.AdamW/v/recommender_net/embedding_1/embeddings:N	J
H
_user_specified_name0.AdamW/m/recommender_net/embedding_1/embeddings:LH
F
_user_specified_name.,AdamW/v/recommender_net/embedding/embeddings:LH
F
_user_specified_name.,AdamW/m/recommender_net/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:FB
@
_user_specified_name(&recommender_net/embedding_3/embeddings:FB
@
_user_specified_name(&recommender_net/embedding_2/embeddings:FB
@
_user_specified_name(&recommender_net/embedding_1/embeddings:D@
>
_user_specified_name&$recommender_net/embedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_embedding_1_layer_call_fn_368238

inputs	
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_368037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368234:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_embedding_2_layer_call_and_return_conditional_losses_368056

inputs	*
embedding_lookup_368047:	�W
identity��embedding_lookup�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_368047inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368047*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_368047*
_output_shapes
:	�W*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^embedding_lookupI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:&"
 
_user_specified_name368047:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_embedding_3_layer_call_and_return_conditional_losses_368280

inputs	*
embedding_lookup_368275:	�W
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_368275inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368275*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name368275:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_embedding_1_layer_call_and_return_conditional_losses_368246

inputs	+
embedding_lookup_368241:
��
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_368241inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368241*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:&"
 
_user_specified_name368241:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_368204c
Orecommender_net_embedding_embeddings_regularizer_l2loss_readvariableop_resource:
��
identity��Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpOrecommender_net_embedding_embeddings_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity8recommender_net/embedding/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: k
NoOpNoOpG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
E__inference_embedding_layer_call_and_return_conditional_losses_368231

inputs	+
embedding_lookup_368222:
��
identity��embedding_lookup�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_368222inputs*
Tindices0	**
_class 
loc:@embedding_lookup/368222*'
_output_shapes
:���������*
dtype0r
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_368222* 
_output_shapes
:
��*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: q
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������~
NoOpNoOp^embedding_lookupG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:&"
 
_user_specified_name368222:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_embedding_layer_call_fn_368219

inputs	
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_368022o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name368215:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_368212d
Qrecommender_net_embedding_2_embeddings_regularizer_l2loss_readvariableop_resource:	�W
identity��Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpQrecommender_net_embedding_2_embeddings_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�W*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity:recommender_net/embedding_2/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
ҙ
�
__inference__traced_save_368410
file_prefixO
;read_disablecopyonread_recommender_net_embedding_embeddings:
��S
?read_1_disablecopyonread_recommender_net_embedding_1_embeddings:
��R
?read_2_disablecopyonread_recommender_net_embedding_2_embeddings:	�WR
?read_3_disablecopyonread_recommender_net_embedding_3_embeddings:	�W,
"read_4_disablecopyonread_iteration:	 0
&read_5_disablecopyonread_learning_rate: Y
Eread_6_disablecopyonread_adamw_m_recommender_net_embedding_embeddings:
��Y
Eread_7_disablecopyonread_adamw_v_recommender_net_embedding_embeddings:
��[
Gread_8_disablecopyonread_adamw_m_recommender_net_embedding_1_embeddings:
��[
Gread_9_disablecopyonread_adamw_v_recommender_net_embedding_1_embeddings:
��[
Hread_10_disablecopyonread_adamw_m_recommender_net_embedding_2_embeddings:	�W[
Hread_11_disablecopyonread_adamw_v_recommender_net_embedding_2_embeddings:	�W[
Hread_12_disablecopyonread_adamw_m_recommender_net_embedding_3_embeddings:	�W[
Hread_13_disablecopyonread_adamw_v_recommender_net_embedding_3_embeddings:	�W+
!read_14_disablecopyonread_total_1: +
!read_15_disablecopyonread_count_1: )
read_16_disablecopyonread_total: )
read_17_disablecopyonread_count: 
savev2_const
identity_37��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead;read_disablecopyonread_recommender_net_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp;read_disablecopyonread_recommender_net_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_1/DisableCopyOnReadDisableCopyOnRead?read_1_disablecopyonread_recommender_net_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp?read_1_disablecopyonread_recommender_net_embedding_1_embeddings^Read_1/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_2/DisableCopyOnReadDisableCopyOnRead?read_2_disablecopyonread_recommender_net_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp?read_2_disablecopyonread_recommender_net_embedding_2_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�W�
Read_3/DisableCopyOnReadDisableCopyOnRead?read_3_disablecopyonread_recommender_net_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp?read_3_disablecopyonread_recommender_net_embedding_3_embeddings^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wd

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	�Wv
Read_4/DisableCopyOnReadDisableCopyOnRead"read_4_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp"read_4_disablecopyonread_iteration^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_learning_rate^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnReadEread_6_disablecopyonread_adamw_m_recommender_net_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpEread_6_disablecopyonread_adamw_m_recommender_net_embedding_embeddings^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_7/DisableCopyOnReadDisableCopyOnReadEread_7_disablecopyonread_adamw_v_recommender_net_embedding_embeddings"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpEread_7_disablecopyonread_adamw_v_recommender_net_embedding_embeddings^Read_7/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_8/DisableCopyOnReadDisableCopyOnReadGread_8_disablecopyonread_adamw_m_recommender_net_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpGread_8_disablecopyonread_adamw_m_recommender_net_embedding_1_embeddings^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_9/DisableCopyOnReadDisableCopyOnReadGread_9_disablecopyonread_adamw_v_recommender_net_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOpGread_9_disablecopyonread_adamw_v_recommender_net_embedding_1_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_10/DisableCopyOnReadDisableCopyOnReadHread_10_disablecopyonread_adamw_m_recommender_net_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpHread_10_disablecopyonread_adamw_m_recommender_net_embedding_2_embeddings^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	�W�
Read_11/DisableCopyOnReadDisableCopyOnReadHread_11_disablecopyonread_adamw_v_recommender_net_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpHread_11_disablecopyonread_adamw_v_recommender_net_embedding_2_embeddings^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wf
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	�W�
Read_12/DisableCopyOnReadDisableCopyOnReadHread_12_disablecopyonread_adamw_m_recommender_net_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpHread_12_disablecopyonread_adamw_m_recommender_net_embedding_3_embeddings^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wf
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�W�
Read_13/DisableCopyOnReadDisableCopyOnReadHread_13_disablecopyonread_adamw_v_recommender_net_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpHread_13_disablecopyonread_adamw_v_recommender_net_embedding_3_embeddings^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�W*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Wf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	�Wv
Read_14/DisableCopyOnReadDisableCopyOnRead!read_14_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp!read_14_disablecopyonread_total_1^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_count_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_16/DisableCopyOnReadDisableCopyOnReadread_16_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpread_16_disablecopyonread_total^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_count^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_37Identity_37:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:%!

_user_specified_namecount:%!

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:NJ
H
_user_specified_name0.AdamW/v/recommender_net/embedding_3/embeddings:NJ
H
_user_specified_name0.AdamW/m/recommender_net/embedding_3/embeddings:NJ
H
_user_specified_name0.AdamW/v/recommender_net/embedding_2/embeddings:NJ
H
_user_specified_name0.AdamW/m/recommender_net/embedding_2/embeddings:N
J
H
_user_specified_name0.AdamW/v/recommender_net/embedding_1/embeddings:N	J
H
_user_specified_name0.AdamW/m/recommender_net/embedding_1/embeddings:LH
F
_user_specified_name.,AdamW/v/recommender_net/embedding/embeddings:LH
F
_user_specified_name.,AdamW/m/recommender_net/embedding/embeddings:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:FB
@
_user_specified_name(&recommender_net/embedding_3/embeddings:FB
@
_user_specified_name(&recommender_net/embedding_2/embeddings:FB
@
_user_specified_name(&recommender_net/embedding_1/embeddings:D@
>
_user_specified_name&$recommender_net/embedding/embeddings:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0	���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�m
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
		user_bias

resto_embedding

resto_bias
	optimizer

signatures"
_tf_keras_model
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
0__inference_recommender_net_layer_call_fn_368136�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
K__inference_recommender_net_layer_call_and_return_conditional_losses_368123�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
!__inference__wrapped_model_368004input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
3
_variables
4_iterations
5_learning_rate
6_index_dict
7
_momentums
8_velocities
9_update_step_xla"
experimentalOptimizer
,
:serving_default"
signature_map
8:6
��2$recommender_net/embedding/embeddings
::8
��2&recommender_net/embedding_1/embeddings
9:7	�W2&recommender_net/embedding_2/embeddings
9:7	�W2&recommender_net/embedding_3/embeddings
�
;trace_02�
__inference_loss_fn_0_368204�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z;trace_0
�
<trace_02�
__inference_loss_fn_1_368212�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z<trace_0
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_recommender_net_layer_call_fn_368136input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_recommender_net_layer_call_and_return_conditional_losses_368123input_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
*__inference_embedding_layer_call_fn_368219�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
�
Etrace_02�
E__inference_embedding_layer_call_and_return_conditional_losses_368231�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_02�
,__inference_embedding_1_layer_call_fn_368238�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
�
Ltrace_02�
G__inference_embedding_1_layer_call_and_return_conditional_losses_368246�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
Rtrace_02�
,__inference_embedding_2_layer_call_fn_368253�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
�
Strace_02�
G__inference_embedding_2_layer_call_and_return_conditional_losses_368265�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zStrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
Ytrace_02�
,__inference_embedding_3_layer_call_fn_368272�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
�
Ztrace_02�
G__inference_embedding_3_layer_call_and_return_conditional_losses_368280�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
_
40
[1
\2
]3
^4
_5
`6
a7
b8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
[0
]1
_2
a3"
trackable_list_wrapper
<
\0
^1
`2
b3"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_368188input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_368204"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_368212"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
N
c	variables
d	keras_api
	etotal
	fcount"
_tf_keras_metric
N
g	variables
h	keras_api
	itotal
	jcount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_embedding_layer_call_fn_368219inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_embedding_layer_call_and_return_conditional_losses_368231inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_embedding_1_layer_call_fn_368238inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_embedding_1_layer_call_and_return_conditional_losses_368246inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_embedding_2_layer_call_fn_368253inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_embedding_2_layer_call_and_return_conditional_losses_368265inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_embedding_3_layer_call_fn_368272inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_embedding_3_layer_call_and_return_conditional_losses_368280inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
>:<
��2,AdamW/m/recommender_net/embedding/embeddings
>:<
��2,AdamW/v/recommender_net/embedding/embeddings
@:>
��2.AdamW/m/recommender_net/embedding_1/embeddings
@:>
��2.AdamW/v/recommender_net/embedding_1/embeddings
?:=	�W2.AdamW/m/recommender_net/embedding_2/embeddings
?:=	�W2.AdamW/v/recommender_net/embedding_2/embeddings
?:=	�W2.AdamW/m/recommender_net/embedding_3/embeddings
?:=	�W2.AdamW/v/recommender_net/embedding_3/embeddings
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count�
!__inference__wrapped_model_368004m0�-
&�#
!�
input_1���������	
� "3�0
.
output_1"�
output_1����������
G__inference_embedding_1_layer_call_and_return_conditional_losses_368246^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0���������
� �
,__inference_embedding_1_layer_call_fn_368238S+�(
!�
�
inputs���������	
� "!�
unknown����������
G__inference_embedding_2_layer_call_and_return_conditional_losses_368265^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0���������
� �
,__inference_embedding_2_layer_call_fn_368253S+�(
!�
�
inputs���������	
� "!�
unknown����������
G__inference_embedding_3_layer_call_and_return_conditional_losses_368280^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0���������
� �
,__inference_embedding_3_layer_call_fn_368272S+�(
!�
�
inputs���������	
� "!�
unknown����������
E__inference_embedding_layer_call_and_return_conditional_losses_368231^+�(
!�
�
inputs���������	
� ",�)
"�
tensor_0���������
� �
*__inference_embedding_layer_call_fn_368219S+�(
!�
�
inputs���������	
� "!�
unknown���������D
__inference_loss_fn_0_368204$�

� 
� "�
unknown D
__inference_loss_fn_1_368212$�

� 
� "�
unknown �
K__inference_recommender_net_layer_call_and_return_conditional_losses_368123f0�-
&�#
!�
input_1���������	
� ",�)
"�
tensor_0���������
� �
0__inference_recommender_net_layer_call_fn_368136[0�-
&�#
!�
input_1���������	
� "!�
unknown����������
$__inference_signature_wrapper_368188x;�8
� 
1�.
,
input_1!�
input_1���������	"3�0
.
output_1"�
output_1���������