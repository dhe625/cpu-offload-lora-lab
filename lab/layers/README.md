# LoRA 연산 최적화 실습: Single-Stream부터 CPU 오프로딩까지 🚀

이 실습에서는 LoRA(Low-Rank Adaptation) 기법을 모델에 적용할 때의 세 가지 다른 실행 전략을 구현합니다. 각 단계를 통해 **CUDA 스트림, 이벤트, CPU-GPU 동기화**와 같은 GPU 프로그래밍의 핵심 개념을 이해하고, 성능 최적화 기법을 직접 코드로 경험하는 것을 목표로 합니다.

## 🎯 실습 목표

1.  **Single-Stream**: 단일 CUDA 스트림을 사용하여 LoRA 연산을 구현합니다.
2.  **Multi-Stream**: 별도의 CUDA 스트림을 사용하여 베이스(Base) 연산과 LoRA 연산을 병렬로 처리하여 성능을 개선합니다.
3.  **CPU**: LoRA 연산을 CPU로 오프로드하고, GPU의 베이스 연산과 동시에 수행하는 기법을 구현합니다.

***

## 📂 실습 1: `lora_single_stream.py` - Single-Stream

가장 기본적인 방식으로, 모든 연산을 단일 CUDA 스트림에서 순차적으로 실행합니다. LoRA 연산의 기본 로직을 이해하는 데 중점을 둡니다.

### 📝 `BaseLayerWithLoRASingleStream` 구현

`forward` 메서드에서 `base_out`에 LoRA 연산 결과를 더해주는 로직을 완성해야 합니다.

* **구현 `TODO`:**
    1.  입력 `x`와 `lora_A` 행렬을 곱합니다. (`x @ self.lora_A.t()`)
    2.  위 결과와 `lora_B` 행렬을 곱하여 `lora_out`을 계산합니다. (`... @ self.lora_B.t()`)
    3.  `base_out`에 `lora_out`을 더합니다 (`.add_()`). LoRA 결과를 `base_out`의 모양(shape)에 맞게 `view`를 사용해 조절해야 할 수 있습니다.


### 📝 `VocabEmbeddingWithLoRASingleStream` 구현

임베딩 레이어에서는 LoRA가 가중치 행렬이 아닌, 조회된 임베딩 벡터에 직접 적용됩니다.

* **구현 `TODO`:**
    1.  입력 `x`에 대해 `lora_A` 임베딩 룩업을 수행합니다. (`F.embedding(x, self.lora_A)`)
    2.  위 결과와 `lora_B` 행렬을 곱합니다.
    3.  `base_out` (원본 임베딩 결과)에 위 결과를 더합니다.


***

## 📂 실습 2: `lora_multi_stream.py` - Multi-Stream

별도의 `lora_stream`을 사용하여 베이스 연산(기본 스트림)과 LoRA 연산(`lora_stream`)을 동시에 실행합니다. **CUDA 이벤트**를 사용한 스트림 간의 동기화가 핵심입니다.

### 📝 `BaseLayerWithLoRAMultiStream` 구현

* **구현 `TODO`:**
    1.  **동기화 시작**: LoRA 연산은 입력 `x`가 준비된 후에 시작되어야 합니다. 현재 스트림에 이벤트를 기록(`self.evt_prev_layer_done.record()`)하고, `lora_stream`이 이 이벤트를 기다리도록(`self.evt_prev_layer_done.wait()`) 설정합니다.
    2.  **LoRA 연산 실행**: `with torch.cuda.stream(self.lora_stream):` 블록 내에서 LoRA 연산을 수행합니다.
    3.  **최종 동기화**: `base_out`에 `lora_out`을 더하기 전에, 기본 스트림이 `lora_stream`의 연산 완료를 기다려야 합니다. `lora_stream`에 `self.evt_lora_done` 이벤트를 기록하고, 기본 스트림이 이 이벤트를 기다리도록 설정합니다.

***

## 📂 실습 3: `lora_cpu.py` - CPU

가장 복잡하고 효율적인 전략입니다. LoRA 연산을 CPU에서 수행하고, 그 결과를 GPU로 비동기적으로 복사하여 GPU의 베이스 연산과 오버랩시킵니다. Prefill과 Decode 단계를 구분하여 최적화합니다.

### 📝 `BaseLayerWithLoRACPU` 구현

* **구현 `TODO` 1: 입력 데이터 비동기 복사**
    * `lora_stream`이 이전 레이어 작업 완료(`cur_stream`)를 기다리게 합니다.
    * `lora_stream` 내에서 GPU 텐서 `x`를 미리 할당된 Pinned Memory `self.x_cpu_v`로 비동기 복사(`copy_`)하고, 복사 완료를 알리는 `self.evt_copy_done` 이벤트를 기록합니다.

* **구현 `TODO` 2: 베이스 연산 완료 기록**
    * `F.linear` 연산 직후, `cur_stream`에 `self.evt_base_done` 이벤트를 기록하여 베이스 연산이 끝났음을 표시합니다.

* **구현 `TODO` 3: Decode 경로 구현**
    * **Prefill** 경로는 이미 구현되어 있습니다. CPU 연산 -> GPU 복사 -> 베이스 연산 대기 -> 덧셈 순서입니다.
    * **Decode** 경로는 GPU 커널을 먼저 호출하여 지연 시간을 최소화하는 것이 목표입니다.
        1.  `lora_stream`에서 커스텀 커널 `cu.wait(self.flag)`을 호출하여 `flag`가 1이 될 때까지 GPU를 대기시킵니다. 그 직후 `base_out`에 `gpu_out_v`를 더하는 연산을 미리 예약합니다.
        2.  `flag_stream`을 사용하여 `evt_copy_done`을 기다린 후, CPU에서 LoRA 연산을 수행하고, 그 결과를 `gpu_out_v`로 비동기 복사합니다.
        3.  모든 복사가 끝나면 `flag_stream`에서 `cu.set_flag(self.flag)`을 호출하여 대기 중이던 `lora_stream`의 `add_` 연산을 깨웁니다.
        4.  `evt_add_done`을 기록합니다.

* **구현 `TODO` 4: 최종 반환 전 동기화**
    * `return` 직전에, `cur_stream`이 `self.evt_add_done`을 기다리도록 하여 모든 연산(베이스 + LoRA)이 완료되었음을 보장합니다.
