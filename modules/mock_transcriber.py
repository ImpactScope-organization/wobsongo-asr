from pathlib import Path
import time
from modules.transcriber import TranscriberProtocol, TranscriptionOutput, ModelType, TargetLanguage

class MockTranscriber(TranscriberProtocol):
    def transcribe(
        self, 
        audio: Path,
        model: ModelType,
        target_lang: TargetLanguage,
        human_transcription: str | None = None,
    ) -> TranscriptionOutput | dict[str, str]:
        
        print(f"Mocking: Processing {audio.name} with model {model.name}...")
        time.sleep(2) 
        
        # Data mock 1 (Moore)
        text_1 = "tɩ paglʋm rɔ sʋka n zemsɩ wa kẽerɩ n tarɩ wɛɛŋɛ tɩ rɔɔ sãn zã dɩgɛ a sẽnna maanɩ bum niŋe wã tɩ dɩkɛ a nuge woto n babnɩ paga pʋgʋm dõm ta zugo maanɩ wa zagma tɩ a dɩkɛ nuge woto n zagma zuga a bãaŋa tɩ yel mõ siiba la sɩkɛ mẽŋa nyẽ mẽŋɛ n loe zu-noogo nyẽ mẽŋɛ kiise bãgrɛ la fãa ya yibgiri ninsaala ka tẽŋ mi yelle la maanɩ la wẽnnɛ fʋ sãn wa yibgi maanɩ wa yel kãŋa n tudgɩrɩ wẽnnɛ sãn tuna tudgɩrɩ tɩ fʋ sãn maanɩ yel kãŋa yaa fõ mõ dʋgɛ n bo nõasɩ ya nõbeegɩrɩ n paasɛ wa paalɛ nẽ siila n zĩkɛ nõbeegɩra wʋ tõm bil fʋ yĩŋe fãa n kʋ dʋgɩ n naagɛ nẽ siila tɩ ba gũusɩ ya gũ vɩɩya tɩ fʋ mẽŋa nyẽ tɩ gũ biisa bee gũurɩ zuo yã tɩ gũ biisi bee gũuri zugu tɩ fʋ tãn dɩkɛ a man doaa nɩ gũuri yella pide woto wẽnnɛ sakre loore poorɩ yel-kãŋa a rʋka fo zugu"

        # Bata moc 2 (Moore-French)
        text_2 = "yam ne ti zomb peelga n dig wam na poughen wa ya tiim har kiik meng meng meng meng menga n songd rapa giina se maaga ti madame suur par noom ne yam yen saad n seosd yam yaar yaare tiim a zoma yam yakda la me n bui ne kamoin bilf bala n sooga n take n dolge minute pistant poughen minute une heure de temps poughen yam na ye a se n teeg n yas bal dãi y suur na y noogo se produit la ce produit là sala c'est pour les hommes qui se sentent faibles vraiment faible madame se plaint prend sala on ne boit pas c'est pas un médicament qu'on consomme on mélange seulement avec du beurre de quarité ou du beurre de cacao et vous massez la partie là avec trente minutes maximum une heure de temps vous allez voir l'effet que ça va faire c'est très très efficace ça tape pas photo vicassa tapa pot aux c"

        # Data Selection Logic based on file name
        if "2" in audio.name:
            dummy_transcript = text_2
            detected_language = "Mooré" 
            print("[MockTranscriber] Using  Data 2 (Mooré-French).")
        else:
            dummy_transcript = text_1
            detected_language = "Mooré"
            print("[MockTranscriber] Using Data 1 (Mooré).")

        dummy_chunks = [
            {"timestamp": [0.0, 5.0], "text": dummy_transcript[:50] + "..."},
            {"timestamp": [5.0, 10.0], "text": "..." + dummy_transcript[50:100]}
        ]
        
        return TranscriptionOutput(
            transcript=dummy_transcript,
            language_selected=detected_language,
            chunks=dummy_chunks,
            final_text=dummy_transcript 
        )