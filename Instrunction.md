# Instrunction

## Dockerの実行手順

1. 画像のビルド  
   ```bash
   make build-image
   ```
   CUDAが利用可能かを自動判定して `gsa:v0` イメージを作成します。

2. コンテナの起動  
   ```bash
   make run
   ```
   初回実行時に `sam_vit_h_4b8939.pth` と `groundingdino_swint_ogc.pth` をカレントディレクトリに自動ダウンロードします。  
   -v "$PWD":/home/appuser/Grounded-Segment-Anything のマウントにより、コンテナ内から編集したファイルはホスト側（ローカル）でそのまま操作できます。

3. 手動でのビルド・起動例  

   ```bash
   docker build --build-arg USE_CUDA=1 --build-arg TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" -t gsa:v0 .
   docker run --gpus all -it --rm --net=host --privileged \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v "$PWD":/home/appuser/Grounded-Segment-Anything \
     -e DISPLAY=$DISPLAY --ipc=host --name gsa gsa:v0
   ```
   CPUのみの場合は --gpus all を外してください。  
   コンテナ内の作業ディレクトリは /home/appuser/Grounded-Segment-Anything です。

## 主なスクリプトの実行

以下はいずれもコンテナ内で実行します。必要に応じて export CUDA_VISIBLE_DEVICES=0 などで使用GPUを指定します。

- GroundingDINO デモ  
  ```bash
  python grounding_dino_demo.py
  ```

- Grounded-SAM デモ  
  ```bash
  python grounded_sam_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image assets/demo1.jpg \
    --output_dir outputs \
    --box_threshold 0.3 --text_threshold 0.25 \
    --text_prompt "bear" --device cuda
  ```

- Grounded-SAM 簡易版  
  ```bash
  python grounded_sam_simple_demo.py
  ```

- Grounded-SAM マルチGPU  
  ```bash
  export CUDA_VISIBLE_DEVICES=0,1
  python grounded_sam_multi_gpu_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_path assets/car --output_dir outputs \
    --box_threshold 0.3 --text_threshold 0.25 \
    --text_prompt "car" --device cuda
  ```

- Inpainting デモ  
  ```bash
  python grounded_sam_inpainting_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image assets/inpaint_demo.jpg \
    --output_dir outputs \
    --box_threshold 0.3 --text_threshold 0.25 \
    --det_prompt "bench" \
    --inpaint_prompt "A sofa, high quality, detailed" \
    --device cuda
  ```

- Gradio アプリ  
  ホスト側で xhost +local: を実行後、  
  ```bash
  python gradio_app.py
  ```
  ブラウザで http://localhost:7860 にアクセスします。

- 自動ラベリング（RAM）  
  ```bash
  git clone https://github.com/xinyu1205/recognize-anything.git
  pip install -r recognize-anything/requirements.txt
  pip install -e recognize-anything/
  python automatic_label_ram_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --ram_checkpoint ram_swin_large_14m.pth \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --input_image assets/demo9.jpg --output_dir outputs \
    --box_threshold 0.25 --text_threshold 0.2 --iou_threshold 0.5 \
    --device cuda
  ```

- OSX 連携デモ  
  ```bash
  git submodule update --init --recursive
  cd grounded-sam-osx && bash install.sh && cd ..
  python grounded_sam_osx_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint groundingdino_swint_ogc.pth \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --osx_checkpoint osx_l_wo_decoder.pth.tar \
    --input_image assets/osx/grounded_sam_osx_demo.png \
    --output_dir outputs \
    --box_threshold 0.3 --text_threshold 0.25 \
    --text_prompt "humans, chairs" --device cuda
  ```

- Chatbot  
  ```bash
  export OPENAI_API_KEY=your_key
  python chatbot.py
  ```

ファイルはコンテナにマウントされたホスト側のディレクトリ内にあるため、ホストで直接編集・保存するとコンテナ内にも反映されます。
