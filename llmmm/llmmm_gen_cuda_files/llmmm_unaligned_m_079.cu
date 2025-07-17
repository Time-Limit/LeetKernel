#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_79_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<79> mm_instantiator;

public:
  UnalignedM_79_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_79_MMInstantiatorWrapper__;

}  // namespace LLMMM
